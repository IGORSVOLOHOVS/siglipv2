"""The script's own code, with the model and the processor replaced: nothing is downloaded."""

from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
import torch
from PIL import Image

import classify_siglip


class FakeInputs(dict[str, Any]):
    """What a processor returns: a mapping that can be moved to a device."""

    def __init__(self) -> None:
        super().__init__(pixel_values=torch.zeros((1, 3, 4, 4)))
        self.device: str | None = None

    def to(self, device: str) -> "FakeInputs":
        self.device = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.inputs = FakeInputs()

    def __call__(self, **kwargs: Any) -> FakeInputs:
        self.calls.append(kwargs)
        return self.inputs


class FakeOutputs:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits_per_image = logits


class FakeModel:
    def __init__(self, logits: list[float]) -> None:
        self.logits = torch.tensor([logits])
        self.received: dict[str, Any] = {}
        self.grad_enabled: bool | None = None

    def __call__(self, **kwargs: Any) -> FakeOutputs:
        self.received = kwargs
        self.grad_enabled = torch.is_grad_enabled()
        return FakeOutputs(self.logits)


def classifier_with(model: FakeModel, processor: FakeProcessor) -> classify_siglip.SigLIPClassifier:
    """A classifier whose loaded model and processor are the fakes."""
    classifier = classify_siglip.SigLIPClassifier.__new__(classify_siglip.SigLIPClassifier)
    classifier.device = "cpu"
    classifier.model = model
    classifier.processor = processor
    return classifier


@pytest.fixture
def image_path(tmp_path: Path) -> str:
    path = tmp_path / "image.png"
    Image.new("L", (8, 8)).save(path)
    return str(path)


def test_the_constructor_loads_the_named_model_and_its_processor_on_the_cpu() -> None:
    loaded = FakeModel([0.0])
    with (
        patch.object(classify_siglip.torch.cuda, "is_available", return_value=False),
        patch.object(classify_siglip.AutoModel, "from_pretrained") as model,
        patch.object(classify_siglip.AutoProcessor, "from_pretrained") as processor,
    ):
        model.return_value.to.return_value = loaded
        classifier = classify_siglip.SigLIPClassifier("some/model")

    model.assert_called_once_with("some/model")
    model.return_value.to.assert_called_once_with("cpu")
    processor.assert_called_once_with("some/model")
    assert classifier.device == "cpu"
    assert classifier.model is loaded


def test_the_constructor_picks_cuda_when_it_is_there() -> None:
    with (
        patch.object(classify_siglip.torch.cuda, "is_available", return_value=True),
        patch.object(classify_siglip.AutoModel, "from_pretrained") as model,
        patch.object(classify_siglip.AutoProcessor, "from_pretrained"),
    ):
        classifier = classify_siglip.SigLIPClassifier()

    model.assert_called_once_with("google/siglip2-base-patch16-224")
    model.return_value.to.assert_called_once_with("cuda")
    assert classifier.device == "cuda"


def test_every_label_becomes_a_prompt_and_the_image_arrives_as_rgb(image_path: str) -> None:
    processor = FakeProcessor()
    classifier_with(FakeModel([0.1, 0.2]), processor).classify(image_path, ["a car", "a cat"])

    (call,) = processor.calls
    assert call["text"] == ["This is a photo of a car.", "This is a photo of a cat."]
    assert call["images"].mode == "RGB"
    assert call["padding"] == "max_length"
    assert call["return_tensors"] == "pt"
    assert processor.inputs.device == "cpu"


def test_the_index_of_the_highest_score_is_returned(image_path: str) -> None:
    model = FakeModel([0.3, 2.5, -1.0])
    index = classifier_with(model, FakeProcessor()).classify(image_path, ["a", "b", "c"])

    assert index == 1
    assert isinstance(index, int)
    assert "pixel_values" in model.received
    assert model.grad_enabled is False


def test_a_missing_image_is_logged_and_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing = str(tmp_path / "missing.png")
    processor = FakeProcessor()

    with pytest.raises(FileNotFoundError):
        classifier_with(FakeModel([0.0]), processor).classify(missing, ["a car"])

    assert f"Image not found: {missing}" in caplog.text
    assert processor.calls == []


class FakeClassifier:
    made_with: ClassVar[list[str]] = []

    def __init__(self, model_name: str) -> None:
        FakeClassifier.made_with.append(model_name)

    def classify(self, image_path: str, labels: list[str]) -> int:
        assert image_path == "data/image.png"
        return labels.index("a truck")


def test_main_prints_the_index_and_the_label(capsys: pytest.CaptureFixture[str]) -> None:
    FakeClassifier.made_with = []
    with patch.object(classify_siglip, "SigLIPClassifier", FakeClassifier):
        classify_siglip.main()

    output = capsys.readouterr().out
    assert "Predicted index: 2" in output
    assert "Predicted label: a truck" in output
    assert FakeClassifier.made_with == ["google/siglip2-base-patch16-224"]


def test_main_reports_a_failure_instead_of_raising(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    with patch.object(
        classify_siglip, "SigLIPClassifier", side_effect=OSError("no weights offline")
    ):
        classify_siglip.main()

    assert "An error occurred: no weights offline" in caplog.text
    assert "Predicted" not in capsys.readouterr().out
