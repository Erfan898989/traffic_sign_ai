"""Run inference on one traffic-sign or traffic-light image."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from traffic_ai.models import build_model
from traffic_ai.vision import IMAGENET_MEAN, IMAGENET_STD, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the class of a single image using a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--top-k", type=int, default=5, help="Number of highest-probability classes to show")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    image_path = Path(args.image)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = checkpoint["class_names"]
    model = build_model(checkpoint["model_name"], checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((int(checkpoint["image_size"]), int(checkpoint["image_size"]))),
        transforms.ToTensor(),
        transforms.Normalize(tuple(checkpoint.get("imagenet_mean", IMAGENET_MEAN)),
                             tuple(checkpoint.get("imagenet_std", IMAGENET_STD))),
    ])
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor).float(), dim=1)[0]
    k = max(1, min(args.top_k, len(class_names)))
    values, indices = torch.topk(probabilities, k=k)

    print(f"Image: {image_path}")
    print(f"Task: {checkpoint.get('dataset', 'unknown')} | Model: {checkpoint['model_name']}")
    print("Predictions:")
    for rank, (prob, idx) in enumerate(zip(values.cpu().tolist(), indices.cpu().tolist()), start=1):
        print(f"  {rank}. {class_names[idx]} — {prob:.2%}")

    if checkpoint.get("dataset") == "traffic_lights":
        print("\nNote: this model predicts the traffic-light state for a cropped/centered light image; "
              "it does not locate lights in a full street scene.")


if __name__ == "__main__":
    main()
