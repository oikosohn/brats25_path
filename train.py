import os
import random
import argparse
import warnings
from glob import glob
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import json


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (matthews_corrcoef, accuracy_score, f1_score, confusion_matrix,
                             roc_auc_score, classification_report, roc_curve)
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle as sk_shuffle

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch.backends.cudnn as cudnn

import timm
from timm.data import resolve_data_config, create_transform
import wandb

from model.virchow2 import Virchow2Classifier
from model.virchow2_peft import Virchow2ClassifierPEFT


with open("label_map.json", "r") as f:
    LABEL_MAP = json.load(f)
    
LABEL_MAP = {int(k): v for k, v in LABEL_MAP.items()}
CLASS_TO_IDX = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(CLASS_TO_IDX)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    # warnings.filterwarnings("ignore")


def autocast_if_amp(args):
    if args.amp_dtype == "fp16":
        dtype = torch.float16
    elif args.amp_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        from contextlib import nullcontext
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def get_grad_scaler(args):
    if args.amp_dtype == "fp16":
        return torch.cuda.amp.GradScaler()
    else:
        return None


def auto_run_name(args):
    today_str = datetime.now().strftime("%Y%m%d-%H%M")
    scheduler_name = "warmrestart" if getattr(args, "warmrestart", False) else "cosine"
    resume_tag = "-resume" if getattr(args, "resume_path", None) else ""
    return (
        f"{today_str}-{args.backbone}-ep{args.num_epochs}-bs{args.batch_size}"
        f"-lr{args.lr}-kfold{args.k_folds}"
        f"-dtype-{args.amp_dtype}-lora_ra-{args.lora_r}_{args.lora_a}"
        f"-sched-{scheduler_name}-{resume_tag}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Virchow2 BRaTS Classification")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="virchow2_peft", choices=["virchow2", "virchow2_peft"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--k_folds", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--train_dir", type=str, default="./input/train")
    parser.add_argument("--valid_dir", type=str, default="./input/valid")
    parser.add_argument("--model_save_dir", type=str, default="./saved_models")
    parser.add_argument("--project_name", type=str, default="brats25path-classification")
    parser.add_argument("--fold_index", type=int, default=-1, help="Specify a particular fold index to run (starts from 0)")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of gradient accumulation step")
    parser.add_argument("--warmrestart", action="store_true", help="Use CosineAnnealingWarmRestarts scheduler instead of CosineAnnealingLR.")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Use label smoothing in CrossEntropyLoss, e.g., 0.1")
    parser.add_argument("--use_aug", action="store_true", help="Apply augmentation transform if specified")

    # LoRA 
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA rank (bottleneck size)")
    parser.add_argument("--lora_a", type=int, default=8, help="LoRA alpha (scaling factor)")

    # AMP
    parser.add_argument("--amp_dtype", type=str, default="float32", choices=["float32", "fp16", "bf16"],
                        help="AMP dtype: float32 (no mixed), fp16, or bf16")

    # Wandb
    parser.add_argument("--wandb_log_misclassed", action="store_true", help="Log misclassified samples to wandb if specified")

    return parser.parse_args()


class NumpyImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = CLASS_TO_IDX

        for cls in sorted(os.listdir(root_dir)):
            if cls not in self.class_to_idx:
                raise ValueError(f"Folder name '{cls}' is not in LABEL_MAP.")

            cls_dir = os.path.join(root_dir, cls)
            files = sorted(glob(os.path.join(cls_dir, "*.npy")))
            for file in files:
                self.samples.append((file, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        np_array = np.load(file_path, mmap_mode='r')

        if self.transform:
            pil_image = Image.fromarray(np_array)
            image = self.transform(pil_image)
        else:
            image = torch.from_numpy(np_array.astype(np.float32)).permute(2, 0, 1) / 255.0
        return image, label, file_path


def main(args):
    if args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 (bf16) is not supported on this GPU.")

    seed_everything(args.seed)

    wandb.login()

    run_name = auto_run_name(args)
    
    # Load default transform
    if args.backbone == "virchow2":
        config = resolve_data_config({}, model="hf-hub:paige-ai/Virchow2")
        backbone_transform = create_transform(**config)
        model = Virchow2Classifier(num_classes=NUM_CLASSES).to(DEVICE)
    elif args.backbone == "virchow2_peft":
        config = resolve_data_config({}, model="hf-hub:paige-ai/Virchow2")
        backbone_transform = create_transform(**config)
        model = Virchow2ClassifierPEFT(lora_r=args.lora_r, lora_alpha=args.lora_a).to(DEVICE)
    else:
        raise ValueError("Unknown backbone.")
    
    if args.use_aug:
        aug_transform = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02)], p=0.5),
            T.RandomApply([T.RandomHorizontalFlip(p=1.0)], p=0.5),
            T.RandomApply([T.RandomVerticalFlip(p=1.0)], p=0.5),
            T.RandomChoice([
                T.RandomRotation(degrees=(0, 0)),
                T.RandomRotation(degrees=(90, 90)),
                T.RandomRotation(degrees=(180, 180)),
                T.RandomRotation(degrees=(270, 270)),
            ])
        ])

        transform = T.Compose([aug_transform, backbone_transform])
        print("[INFO] Augmentation transform applied.")
    else:
        transform = backbone_transform
        print("[INFO] No augmentation transform applied.")

    train_dataset = NumpyImageDataset(args.train_dir, transform=transform)
    test_dataset = NumpyImageDataset(args.valid_dir, transform=transform)
    
    full_dataset = train_dataset
    full_dataset.samples += test_dataset.samples
    labels = [label for _, label in full_dataset.samples]

    if args.max_samples:
        if args.max_samples < NUM_CLASSES:
            raise ValueError(f"--max_samples must be >= number of classes ({NUM_CLASSES})")
        
        label_to_samples = defaultdict(list)
        for sample, label in full_dataset.samples:
            label_to_samples[label].append(sample)
            
        selected_samples, selected_labels = [], []
        per_class = args.max_samples // NUM_CLASSES
        for label, samples_list in label_to_samples.items():
            chosen = random.sample(samples_list, min(per_class, len(samples_list)))
            selected_samples.extend(chosen)
            selected_labels.extend([label] * len(chosen))
            
        selected_samples, selected_labels = sk_shuffle(selected_samples, selected_labels, random_state=args.seed)
        full_dataset.samples = list(zip(selected_samples, selected_labels))
        labels = selected_labels
    
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, valid_idx) in tqdm(enumerate(skf.split(full_dataset.samples, labels)), total=args.k_folds, desc="K-Fold Training"):
        if args.fold_index is not None and args.fold_index >= 0 and fold != args.fold_index:
            continue
                
        run_name_fold = f"{run_name}-fold{fold+1}"
        wandb.init(project=args.project_name, name=run_name_fold, config=vars(args)) 

        train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers, prefetch_factor=4, pin_memory=True)
        valid_loader = DataLoader(Subset(full_dataset, valid_idx), batch_size=args.batch_size, shuffle=False, 
                                  num_workers=args.num_workers, prefetch_factor=4, pin_memory=True)

        # Initialize the model
        if args.backbone == "virchow2":
            model = Virchow2Classifier(num_classes=NUM_CLASSES, img_size=args.img_size).to(DEVICE)
        elif args.backbone == "virchow2_peft":
            model = Virchow2ClassifierPEFT(num_classes=NUM_CLASSES, 
                                           img_size=args.img_size,
                                           lora_r=args.lora_r,
                                           lora_alpha=args.lora_a,).to(DEVICE)
        
        # Define the loss function
        if args.label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            print(f"[INFO] Using CrossEntropyLoss with label_smoothing={args.label_smoothing}")
        else:
            criterion = nn.CrossEntropyLoss()
            print(f"[INFO] Using standard CrossEntropyLoss (no smoothing)")
        
        # Define the optimizer and scheduler
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        if args.warmrestart:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=args.num_epochs // 2, eta_min=0
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.num_epochs, eta_min=0
            )

        run_save_dir = os.path.join(args.model_save_dir, run_name)
        os.makedirs(run_save_dir, exist_ok=True)

        # Resume training from a checkpoint
        start_epoch = 0
        if args.resume_path and os.path.exists(args.resume_path):
            checkpoint = torch.load(args.resume_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"\n[INFO] Resuming from epoch {start_epoch}...\n")

        # K-Fold Epoch Loop
        scaler = get_grad_scaler(args)

        for epoch in range(start_epoch, args.num_epochs):
            # ========== Train ==========            
            model.train()
            train_loss = 0.0
            train_samples = 0
            train_preds, train_labels = [], []

            for step, (imgs, labels_batch, _) in enumerate(tqdm(train_loader, 
                                                                desc=f"[Fold {fold+1}] Epoch {epoch+1} - Train", leave=False)):
                imgs, labels_batch = imgs.to(DEVICE), labels_batch.to(DEVICE)

                if step % args.grad_accum_steps == 0:
                    # Reset gradients at the start of accumulation
                    optimizer.zero_grad() 
                    
                with autocast_if_amp(args):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels_batch)
                    
                    batch_size = imgs.size(0)
                    train_loss += loss.item() * batch_size
                    train_samples += batch_size
                    
                    # Scale loss for gradient accumulation
                    loss = loss / args.grad_accum_steps
      
                if scaler is not None:
                    # Backward on scaled loss for AMP
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                # Optimizer step on accumulation
                if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    if scaler is not None:
                        # Unscale gradients and update optimizer
                        scaler.step(optimizer)
                        
                        # Adjust scaling factor for next iteration based on overflow
                        scaler.update()
                        
                    else:
                        optimizer.step()           
                                     
                    # collect predictions for train metrics
                    probs = torch.softmax(outputs, dim=1).float().detach().cpu().numpy()
                    preds = np.argmax(probs, axis=1)
                    train_preds.extend(preds)
                    train_labels.extend(labels_batch.cpu().numpy())
                                                            
            avg_train_loss = train_loss / train_samples
            
            # compute train metrics
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average="macro")
            train_mcc = matthews_corrcoef(train_labels, train_preds)

            # ========== Valid ==========
            model.eval()
            val_preds, val_labels, val_probs, val_paths = [], [], [], []
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for imgs, labels_batch, paths in tqdm(valid_loader, desc=f"[Fold {fold+1}] Epoch {epoch+1} - Valid", leave=False):
                    imgs, labels_batch = imgs.to(DEVICE), labels_batch.to(DEVICE)
                    with autocast_if_amp(args):
                        outputs = model(imgs)
                        loss = criterion(outputs, labels_batch)

                    batch_size = imgs.size(0)
                    
                    # Mini-batch mean * number of samples
                    val_loss += loss.item() * batch_size   
                    val_samples += batch_size
                    
                    probs = torch.softmax(outputs, dim=1).float().cpu().numpy()
                    preds = np.argmax(probs, axis=1)
                    val_probs.extend(probs)
                    val_preds.extend(preds)
                    val_labels.extend(labels_batch.cpu().numpy())
                    val_paths.extend(paths)
                    
            avg_val_loss = val_loss / val_samples
            
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average="macro")
            val_mcc = matthews_corrcoef(val_labels, val_preds)

            # save checkpoint
            epoch_save_path = os.path.join(run_save_dir, f"epoch{epoch+1}_fold{fold+1}.pt")
            print(f'Path to save the trained model : {epoch_save_path}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, epoch_save_path)

            # ========== Logging ==========
            cm = confusion_matrix(val_labels, val_preds)
            fig_cm, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=LABEL_MAP.values(), yticklabels=LABEL_MAP.values())
            plt.title("Confusion Matrix")
            wandb.log({"confusion_matrix": wandb.Image(fig_cm)})
            plt.close(fig_cm)

            bin_labels = label_binarize(val_labels, classes=list(range(NUM_CLASSES)))
            val_probs_arr = np.array(val_probs)
            try:
                auc_macro = roc_auc_score(bin_labels, val_probs_arr, average="macro", multi_class="ovr")
                fig_roc, ax = plt.subplots(figsize=(8, 6))
                
                for i in range(NUM_CLASSES):
                    fpr, tpr, _ = roc_curve(bin_labels[:, i], val_probs_arr[:, i])
                    ax.plot(fpr, tpr, label=f"{LABEL_MAP[i]}")
                    
                ax.plot([0, 1], [0, 1], "k--")
                ax.set_title(f"Per-Class ROC Curve (AUC={auc_macro:.4f})")
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.legend()
                wandb.log({"roc_auc_macro": auc_macro, "roc_curve": wandb.Image(fig_roc)})
                plt.close(fig_roc)

            except ValueError as e:
                auc_macro = None
                print(f"[ROC ERROR] {e}")

            misclassified = [(p, t, s) for p, t, s in zip(val_preds, val_labels, val_paths) if p != t]
            table = wandb.Table(columns=["pred", "true", "path"])
            for pred, true, path in misclassified:
                table.add_data(LABEL_MAP[pred], LABEL_MAP[true], path)
                
            if args.wandb_log_misclassed:
                wandb.log({"misclassified_samples": table})
                
            misclassified_df = pd.DataFrame(
                [
                    {
                        "pred": LABEL_MAP[p],
                        "true": LABEL_MAP[t],
                        "path": s
                    }
                    for p, t, s in misclassified
                ]
            )

            # Save misclassified samples to CSV
            misclassified_path = os.path.join(run_save_dir, f"misclassified_epoch{epoch+1}_fold{fold+1}.csv")
            misclassified_df.to_csv(misclassified_path, index=False)

            print(f"[INFO] Misclassified samples saved to {misclassified_path}")

            report_table = wandb.Table(columns=["class", "precision", "recall", "f1-score", "support"])
            report = classification_report(val_labels, val_preds, target_names=list(LABEL_MAP.values()), 
                                           output_dict=True, zero_division=0)
            for class_name in LABEL_MAP.values():
                if class_name in report:
                    row = report[class_name]
                    report_table.add_data(
                        class_name,
                        row["precision"],
                        row["recall"],
                        row["f1-score"],
                        row["support"]
                    )
            wandb.log({
                "classification_report": report_table,
                "train_acc/epoch": train_acc,
                "train_f1/epoch": train_f1,
                "train_mcc/epoch": train_mcc,
                "train_loss/epoch": avg_train_loss,
                "valid_loss/epoch": avg_val_loss,
                "val_acc/epoch": val_acc,
                "val_f1/epoch": val_f1,
                "val_mcc/epoch": val_mcc,
                "epoch": epoch + 1,
                "lr/epoch": scheduler.get_last_lr()[0]
            })
            
            print(f"[Fold {fold+1}] Epoch {epoch+1} Metrics:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, MCC: {train_mcc:.4f}")
            print(f"  Val   Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, MCC: {val_mcc:.4f}")
            
            if 'auc_macro' in locals():
                print(f"  Val AUC Macro: {auc_macro:.4f}")
            
            print(f"  LR/EPOCH: {scheduler.get_last_lr()[0]:.4f}")

            cm_df = pd.DataFrame(cm, index=LABEL_MAP.values(), columns=LABEL_MAP.values())
            print("[Confusion Matrix]")
            print(cm_df)
            print("[Classification Report]")
            print(report)
            
            scheduler.step()
            
        wandb.finish()

    print("Training complete.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
