import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
from transformers import AutoModel


CONFIG = {
    "device": "cuda:2" if torch.cuda.is_available() else "cpu",
    "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "img_size": 2400,
    "patch_size": 16,
    "num_classes": 8,
    "extract_layers": [-1],
    "checkpoint_path": "./training_result/7classes_vitl_layer-1_LandR_v3_argumented/best.pth",
    "test_image_dir": "./Inference/test_datasets",
    "output_dir": "./Inference/petal_only_outputs",
    "target_class_ids": [6, 7],
}


class FineTuneSegmentation(nn.Module):
    def __init__(self, model_name, num_classes, extract_layers):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.extract_layers = extract_layers

        self.embed_dim = self.backbone.config.hidden_size
        self.total_dim = self.embed_dim * len(extract_layers)

        self.head = nn.Sequential(
            nn.BatchNorm2d(self.total_dim),
            nn.Conv2d(self.total_dim, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

    def forward(self, x):
        batch_size, _, height, width = x.shape
        outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        features = []
        num_patches = (height // CONFIG["patch_size"]) * (width // CONFIG["patch_size"])
        patch_h, patch_w = height // CONFIG["patch_size"], width // CONFIG["patch_size"]

        for layer_idx in self.extract_layers:
            feat = hidden_states[layer_idx]
            feat_spatial = feat[:, -num_patches:, :].permute(0, 2, 1)
            feat_spatial = feat_spatial.reshape(batch_size, self.embed_dim, patch_h, patch_w)
            features.append(feat_spatial)

        cat_feats = torch.cat(features, dim=1)
        logits = self.head(cat_feats)
        return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)


def build_transform(img_size):
    return v2.Compose(
        [
            v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def infer_mask(model, transform, image_pil, device):
    original_width, original_height = image_pil.size
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    return cv2.resize(pred_mask.astype(np.uint8), (original_width, original_height), interpolation=cv2.INTER_NEAREST)


def save_petal_outputs(original_pil, pred_mask, output_root, image_name, target_class_ids):
    base_name = os.path.splitext(image_name)[0]

    petal_class_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    for class_id in target_class_ids:
        petal_class_mask[pred_mask == class_id] = class_id

    binary_mask = np.isin(pred_mask, target_class_ids).astype(np.uint8) * 255

    original_rgb = np.array(original_pil)
    black_bg_rgb = np.zeros_like(original_rgb)
    black_bg_rgb[binary_mask > 0] = original_rgb[binary_mask > 0]
    alpha_channel = binary_mask
    cutout_rgba = np.dstack([original_rgb, alpha_channel])

    mask_dir = os.path.join(output_root, "petal_masks")
    class_mask_dir = os.path.join(output_root, "petal_class_masks")
    black_bg_dir = os.path.join(output_root, "petal_black_bg_rgb")
    cutout_dir = os.path.join(output_root, "petal_cutouts")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(class_mask_dir, exist_ok=True)
    os.makedirs(black_bg_dir, exist_ok=True)
    os.makedirs(cutout_dir, exist_ok=True)

    Image.fromarray(binary_mask, mode="L").save(os.path.join(mask_dir, f"{base_name}_petal_mask.png"))
    Image.fromarray(petal_class_mask, mode="L").save(os.path.join(class_mask_dir, f"{base_name}_petal_class_mask.png"))
    Image.fromarray(black_bg_rgb, mode="RGB").save(os.path.join(black_bg_dir, f"{base_name}_petal_black_bg.png"))
    Image.fromarray(cutout_rgba, mode="RGBA").save(os.path.join(cutout_dir, f"{base_name}_petal_cutout.png"))


def parse_args():
    parser = argparse.ArgumentParser(description="Extract class 6 and 7 pixels for petal color correction.")
    parser.add_argument("--test-image-dir", default=CONFIG["test_image_dir"])
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    parser.add_argument("--checkpoint-path", default=CONFIG["checkpoint_path"])
    parser.add_argument("--device", default=CONFIG["device"])
    parser.add_argument("--img-size", type=int, default=CONFIG["img_size"])
    parser.add_argument("--class-ids", nargs="+", type=int, default=CONFIG["target_class_ids"])
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = FineTuneSegmentation(CONFIG["model_name"], CONFIG["num_classes"], CONFIG["extract_layers"]).to(args.device)

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found: {args.checkpoint_path}")
        return

    print(f"Loading fine-tuned weights from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint)
    model.eval()

    transform = build_transform(args.img_size)

    test_files = [f for f in os.listdir(args.test_image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Total images: {len(test_files)}")

    for image_name in test_files:
        image_path = os.path.join(args.test_image_dir, image_name)
        try:
            original_pil = Image.open(image_path).convert("RGB")
        except Exception as exc:
            print(f"Skip {image_name}: {exc}")
            continue

        pred_mask = infer_mask(model, transform, original_pil, args.device)
        save_petal_outputs(original_pil, pred_mask, args.output_dir, image_name, args.class_ids)
        print(f"Processed: {image_name}")


if __name__ == "__main__":
    main()