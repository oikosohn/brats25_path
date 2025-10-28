FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

ENV HF_HUB_OFFLINE=1
ENV HF_HOME=/workspace/model

RUN apt-get update -y

COPY requirements.txt .
COPY label_map.json .

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy local Hugging Face model to hf_model/ (used when offline, no internet)
# COPY hf_model hf_model/

# Copy input folder containing PNG and JPEG images
COPY input input/

# Copy checkpoints for inference
COPY checkpoints checkpoints/

COPY model model/
COPY infer_worker.py .
COPY infer_main.py .
COPY train.py .

# To run training
# CMD ["python", "train.py", "--device_ids", "0", "--backbone", "virchow2_peft", "--batch_size", "16", "--img_size", "224", "--max_epochs", "20", "--output_dir", "checkpoints/virchow2_peft"]

# To run inference on images in /input directory and save results to /output directory
# CMD ["python", "infer_main.py", "--device_ids", "0", "--backbone", "virchow2_peft", "--tta", "4"]

