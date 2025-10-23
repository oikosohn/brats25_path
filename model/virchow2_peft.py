import torch
import torch.nn as nn
import timm
from peft import LoraConfig, get_peft_model


class Virchow2ClassifierPEFT(nn.Module):
    def __init__(self, num_classes=9, lora_r=4, lora_alpha=8, freeze_backbone=True, img_size=224):
        super().__init__()
        
        # 1. backbone 
        self.backbone = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=nn.SiLU,
            num_classes=0,
        )
        
        # 2. classifier head
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            out = self.backbone(dummy)
            cls_dim = out[:, 0].shape[1]
            patch_dim = out[:, 1:].shape[2]
            self.embedding_dim = cls_dim + patch_dim
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

        # 3. LoRA Config
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=r"blocks\.\d+\.attn\.qkv",  # train attn.qkv module in each block
            modules_to_save=["classifier"],  # keep the head (classifier) without LoRA training
        )

        # 4. PEFT model wrapping
        self.backbone = get_peft_model(self.backbone, lora_config)

        # 5. backbone freeze: train only LoRA and classifier
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if ("lora_" in name) or ("classifier" in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for name, param in self.classifier.named_parameters():
                param.requires_grad = True

        return self.backbone.print_trainable_parameters()
        
    def forward(self, x):
        x = self.backbone(x)
        class_token = x[:, 0]
        patch_tokens = x[:, 1:]
        embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=1)
        return self.classifier(embedding)