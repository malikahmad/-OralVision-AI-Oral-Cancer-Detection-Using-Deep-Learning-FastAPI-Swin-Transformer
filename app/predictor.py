
from pathlib import Path

import timm # type: ignore
import torch
from PIL import Image
from torchvision import transforms


class SwinPredictor:
    def __init__(self, checkpoint_path: Path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.transform = None
        self.idx_to_label = {}
        self.error = None

        if not checkpoint_path.exists():
            self.error = (
                f"Model not found: {checkpoint_path}. "
                "Copy oral_swin_model.pth into the models folder."
            )
            return

        try:
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu"
                )

            class_to_idx = checkpoint["class_to_idx"]
            label_map = checkpoint["label_map"]

            self.idx_to_label = {
                int(class_to_idx[name]): label_map[name]
                for name in class_to_idx
            }

            self.model = timm.create_model(
                checkpoint["model_name"],
                pretrained=False,
                num_classes=2,
            )

            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.to(self.device)
            self.model.eval()

            size = int(checkpoint.get("img_size", 224))
            mean = checkpoint.get(
                "mean", [0.485, 0.456, 0.406]
            )
            std = checkpoint.get(
                "std", [0.229, 0.224, 0.225]
            )

            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

        except Exception as exc:
            self.error = f"Unable to load model: {exc}"

    @property
    def ready(self):
        return self.model is not None and self.error is None

    def predict(self, image: Image.Image):
        if not self.ready:
            raise RuntimeError(self.error or "Model unavailable.")

        tensor = self.transform(
            image.convert("RGB")
        ).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        probabilities = probabilities.cpu()
        index = int(probabilities.argmax().item())

        scores = {
            self.idx_to_label[i]: round(
                float(probabilities[i]) * 100, 2
            )
            for i in range(len(probabilities))
        }

        return {
            "prediction": self.idx_to_label[index],
            "confidence": round(
                float(probabilities[index]) * 100, 2
            ),
            "scores": scores,
            "device": str(self.device),
        }
