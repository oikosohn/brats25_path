import torch
import torch.nn as nn
import timm


class Virchow2Classifier(nn.Module):
    def __init__(self, num_classes=9, freeze_backbone=True, img_size=224):
        super().__init__()
        self.backbone = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            num_classes=0,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(1280 * 2, num_classes)

    def forward(self, x):
        x = self.backbone(x)           
        class_token = x[:, 0]          
        patch_tokens = x[:, 5:]        
        patch_token_mean = patch_tokens.mean(dim=1) 
        embedding = torch.cat([class_token, patch_token_mean], dim=-1) 
        out = self.classifier(embedding)
        return out