import argparse
import os
import json

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image, ImageFile
import timm
import pandas as pd
from tqdm import tqdm


from model.virchow2_peft import Virchow2ClassifierPEFT

ImageFile.LOAD_TRUNCATED_IMAGES = True

with open("label_map.json", "r") as f:
    LABEL_MAP = json.load(f)
    
LABEL_MAP = {int(k): v for k, v in LABEL_MAP.items()}

NUM_CLASSES = len(LABEL_MAP)

def get_tta_transforms(img_size, tta):
    transforms_list = []
    base = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    
    if tta == 1:
        transforms_list.append(transforms.Compose(base))
    elif tta == 4:
        for hflip, vflip in [(False, False), (True, False), (False, True), (True, True)]:
            aug = []
            if hflip: aug.append(transforms.RandomHorizontalFlip(p=1.0))
            if vflip: aug.append(transforms.RandomVerticalFlip(p=1.0))
            transforms_list.append(transforms.Compose(aug + base))
    elif tta == 8:
        for rot in [0, 90, 180, 270]:
            for hflip in [False, True]:
                aug = []
                if rot != 0: aug.append(transforms.RandomRotation(degrees=(rot, rot)))
                if hflip: aug.append(transforms.RandomHorizontalFlip(p=1.0))
                transforms_list.append(transforms.Compose(aug + base))
    elif tta == 12:
        for rot in [0, 90, 180, 270]:
            for hflip in [False, True]:
                aug = []
                if rot != 0: aug.append(transforms.RandomRotation(degrees=(rot, rot)))
                if hflip: aug.append(transforms.RandomHorizontalFlip(p=1.0))
                transforms_list.append(transforms.Compose(aug + base))
    else:
        transforms_list.append(transforms.Compose(base))
        
    return transforms_list


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.jpg', '.png'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        filename = os.path.basename(img_path)
        
        return image, filename


def run_single_inference(args):
    
    backbone = args.backbone
    image_dir = args.image_dir
    model_path = args.model_path
    output_probs = args.output_probs
    img_size = args.img_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    device = args.device
    tta = args.tta
    folder_path = args.folder_path
    amp_dtype = getattr(args, 'amp_dtype', 'float32') 

    tta_transforms = get_tta_transforms(img_size, tta)
    if backbone == "virchow2_peft":
        model = Virchow2ClassifierPEFT(num_classes=NUM_CLASSES, freeze_backbone=False, img_size=img_size)
    else:
        raise ValueError()
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"[Error] Failed to load model: {model_path}, {e}")
        raise
    
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Set AMP dtype
    if amp_dtype == "bf16":
        autocast_dtype = torch.bfloat16
    elif amp_dtype == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    all_probs = []
    all_filenames = None

    if isinstance(device, str) and device.startswith("cuda:"):
        device_num = int(device.split(":")[1])
    else:
        device_num = 0

    for tta_idx, ttf in enumerate(tta_transforms):
        dataset = ImageDataset(image_dir, transform=ttf)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=4)
        probs_list = []
        filenames_list = []
        
        with torch.no_grad():
            for images, filenames in tqdm(
                dataloader,
                desc=f"TTA-{tta_idx+1}/{len(tta_transforms)} | {os.path.basename(model_path)} | GPU:{device_num}",
                position=device_num,
                leave=True
            ):
                images = images.to(device)
                if autocast_dtype:
                    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                        logits = model(images)
                        probs = torch.softmax(logits, dim=1)
                else:
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)
                probs_list.append(probs.cpu())
                filenames_list.extend(filenames)
                
        probs = torch.cat(probs_list, dim=0)
        all_probs.append(probs)
        
        if all_filenames is None:
            all_filenames = filenames_list
        else:
            assert all_filenames == filenames_list, "Order of filenames differs between TTA runs!"

    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    preds = torch.argmax(avg_probs, dim=1).numpy()
    filenames = all_filenames

    df = pd.DataFrame(
        avg_probs.numpy(),
        columns=[f"prob_{i}" for i in range(avg_probs.shape[1])]
    )
    df.insert(0, "SubjectID", filenames)        
    df["Prediction"] = preds                    
    
    df = df.sort_values("SubjectID").reset_index(drop=True)
    df.to_csv(output_probs, index=False)
    print(f"✅ saved per-model predictions to {output_probs}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="virchow2_peft", choices=["virchow2", "virchow2_peft"])
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True) 
    parser.add_argument('--output_probs', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tta', type=int, default=1)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--folder_path', type=str, required=True)
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['float32', 'fp16', 'bf16'])

    args = parser.parse_args()

    if args.worker:
        run_single_inference(args)

if __name__ == '__main__':
    main()
