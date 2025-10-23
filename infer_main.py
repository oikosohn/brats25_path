import argparse
import subprocess
import torch
import pandas as pd
import os

import numpy as np

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="virchow2_peft", choices=["virchow2", "virchow2_peft"])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--image_dir', type=str, default='/input/test')
    parser.add_argument('--model_paths', type=str, default='checkpoints/epoch50_fold1.pt,checkpoints/epoch50_fold2.pt,checkpoints/epoch50_fold3.pt,checkpoints/epoch50_fold4.pt', \
                        help='Comma-separated list of .pt model paths')
    parser.add_argument('--device_ids', type=str, required=True, help='Comma-separated list of device IDs, e.g. 0,1,2')
    parser.add_argument('--output_csv', type=str, default=f'/output/submission.csv')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--tta', type=int, default=1, help='Number of Test Time Augmentation runs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["float32", "fp16", "bf16"],
                        help="AMP dtype: float32 (no mixed), fp16, or bf16")
    args = parser.parse_args()

    model_paths = [p.strip() for p in args.model_paths.split(",")]
    device_ids = [d.strip() for d in args.device_ids.split(",")]
    
    if len(device_ids) == 1:
        device_ids = device_ids * len(model_paths)
    elif len(device_ids) != len(model_paths):
        raise ValueError(
            "`device_ids` must either contain a single element or match the number of `model_paths`."
        )

    sequential = len(set(device_ids)) == 1         

    prob_files = []
    processes = []
    folder_name = '/output/probs'
    os.makedirs(folder_name, exist_ok=True)
    
    for idx, (model_path, device_id) in enumerate(zip(model_paths, device_ids)):        
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        prob_file = f"{folder_name}/prob_{model_name}.csv"
        prob_files.append(prob_file)
        cmd = [
            "python", "-u", "infer_worker.py",
            "--backbone", args.backbone,
            "--image_dir", args.image_dir,
            "--model_path", model_path,
            "--output_probs", prob_file,
            "--img_size", str(args.img_size),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--device", f"cuda:{device_id}",
            "--tta", str(args.tta),
            "--worker", 
            "--folder_path", folder_name,
            "--amp_dtype", args.amp_dtype,
        ]
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        mode = "[SEQ]" if sequential else "[PAR]"
        print(f"{mode}  {model_name} on cuda:{device_id}")
        print(f"{' '.join(cmd)}")

        if sequential:
            subprocess.run(cmd, check=True, env=env)
        else:                              
            processes.append(subprocess.Popen(cmd))

    for proc in processes:
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Worker (PID {proc.pid}) exited with status {ret}")

    num_classes = args.num_classes
    all_probs = []
    for p in prob_files:
        df = pd.read_csv(p)
        probs = df[[f"prob_{i}" for i in range(num_classes)]].to_numpy() 
        all_probs.append(probs)
        
    avg_probs = np.mean(np.stack(all_probs), axis=0)
    preds = np.argmax(avg_probs, axis=1)
    filenames = df["SubjectID"].tolist()

    submission = pd.DataFrame({
        "SubjectID": filenames,
        "Prediction": preds
    })
    
    submission = submission.sort_values("SubjectID").reset_index(drop=True)
    
    submission.to_csv(args.output_csv, index=False)
    print(f"✅ submission saved to {args.output_csv}")


if __name__ == '__main__':
    main()
