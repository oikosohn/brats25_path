## Patch-level Classification of Histopathological Subregions in Glioblastoma

This is a PyTorch implementation of "Patch-level Classification of Histopathological Subregions in Glioblastoma"

This work won 3rd place in the BraTS Path Challenge at MICCAI 2025.

## Usage

### Build the Docker Image

```bash
docker build -t virchow2_image .
```

### Run Training

```bash
docker run --gpus all \
    -v /path/to/input:/input \
    -v /path/to/saved_models:/saved_models \
    virchow2_image \
    python train.py \
        --device_ids 0 \
        --backbone virchow2_peft \
        --batch_size 16 \
        --img_size 224 \
        --num_epochs 50 \
        --train_dir /input/train \
        --valid_dir /input/valid \
        --model_save_dir /saved_models \
        --fold_index -1 \
        --use_aug
```

### Run Inference

```bash
docker run --gpus all \
    -v /path/to/input:/input \
    -v /path/to/output:/output \
    virchow2_image \
    python infer_main.py \
        --device_ids 0 \
        --backbone virchow2_peft \
        --tta 4
```

## License

This project is licensed under [CC BY-NC-ND](https://github.com/oikosohn/brats25_path/blob/main/LICENSE-CC-BY-NC-ND).

## BibTeX

TBD
