import logging
from typing import Protocol

import torch
from PIL import Image
from rich.console import Console
from transformers import AutoModel, AutoProcessor

console = Console()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Classifier(Protocol):
    def classify(self, image_path: str, labels: list[str]) -> int: ...


class SigLIPClassifier:
    """
    SigLIP-based image classifier for zero-shot tasks.
    """

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model {model_name} on {self.device}...")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def classify(self, image_path: str, labels: list[str]) -> int:
        """
        Classifies an image against a list of labels using SigLIP 2.
        Returns the index of the most probable text embedding.
        """
        # Applying prompt template as per SigLIPv2 best practices
        prompts = [f"This is a photo of {label}." for label in labels]

        logger.info(f"Classifying {image_path}")
        logger.debug(f"Using prompts: {prompts}")

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"Image not found: {image_path}")
            raise

        inputs = self.processor(
            text=prompts, images=image, padding="max_length", return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # SigLIP returns logits_per_image (alignment scores)
        logits_per_image = outputs.logits_per_image
        best_index = int(torch.argmax(logits_per_image, dim=-1).item())

        return best_index


def main() -> None:
    # Task configuration
    image_path = "data/image.png"
    candidate_labels = ["a car", "a bicycle", "a truck"]
    model_name = "google/siglip2-base-patch16-224"

    # Inference
    try:
        classifier = SigLIPClassifier(model_name)
        index = classifier.classify(image_path, candidate_labels)

        console.print(f"[bold green]Predicted index: {index}[/bold green]")
        console.print(f"[bold green]Predicted label: {candidate_labels[index]}[/bold green]")

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
