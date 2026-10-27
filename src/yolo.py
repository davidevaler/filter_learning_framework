"""
yolo.py
Wrapper for YOLO model operations including loading, evaluation, and configuration.
"""

from pathlib import Path
from typing import List, Literal
from typing_extensions import Self, deprecated

from pydantic import BaseModel, Field
from ultralytics.models import YOLO

from .dataset import (
    Dataset,
    DatasetSplit,
)

# Order of IDs as in the COCO dataset files (resources/dataset/coco_annotations)
CLASS_NAMES = ["schizont", "gametocyte", "ring", "trophozoite"]


def free_gpu_memory() -> None:
    """Free the GPU memory occupied by unused objects."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class YoloConfig(BaseModel):
    path: Path = Field(..., description="Dataset root path.")
    train: str = Field(default="images", description="Training data directory.")
    test: str = Field(default="images", description="Testing data directory.")
    val: str = Field(default="images", description="Validation data directory.")
    nc: int = Field(default=4, description="Number of classes.")
    names: List[str] = Field(
        default_factory=lambda: list(CLASS_NAMES),
        description="Names of the classes.",
    )

    def store_yaml(self) -> Path:
        config_path = self.path / "yolo_config.yaml"
        config_path.write_text(self.model_dump_yaml())
        return config_path

    def model_dump_yaml(self) -> str:
        import yaml

        dump = self.model_dump()
        dump["path"] = dump["path"].as_posix()
        return yaml.dump(dump, default_flow_style=False)


class Yolo:
    """
    Wrapper for YOLO model operations including loading, evaluation, and configuration.
    """

    model_path: Path
    yolo_model: YOLO
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    conf: float = 0.25

    @deprecated("Yolo class should be instantiated via 'load_model' method.")
    def __init__(self, *_, **__):
        ...

    @classmethod
    def load_model(
        cls,
        model_path: Path,
        device: Literal["cpu", "mps", "cuda"] = "cpu",
        conf: float = 0.25,
    ) -> Self:
        """
        conf: confidence threshold used in evaluate. 0.001 is the standard value
        for calculating mAP; 0.25 is the value originally used by the framework.
        """

        if not model_path.exists():
            raise Exception(f"Given YOLO path doesn't exist: {model_path}")

        instance = object.__new__(cls)
        instance.model_path = model_path
        instance.yolo_model = YOLO(model_path.as_posix())
        instance.device = device
        instance.conf = conf

        return instance

    def evaluate(self, dataset: Dataset) -> float:
        """
        Evaluates the YOLO model on a temporary dataset.
        Returns the mAP@0.5 metric.
        """
        from ultralytics.utils.metrics import DetMetrics

        if not dataset.samples:
            raise ValueError("Cannot evaluate an empty dataset.")

        # Create a temporary YOLO configuration file for evaluation
        dataset_root = getattr(dataset, "base_path", None)
        if dataset_root is None:
            # if the path is, for example, /tmp/.../yolo_eval_dataset/images/xxx.png,
            # the root of the YOLO dataset is /tmp/.../yolo_eval_dataset
            dataset_root = dataset.samples[0].image_path.parent.parent

        config = YoloConfig(
            path=Path(dataset_root),
            names=list(CLASS_NAMES),
            nc=4
        )
        config_path = config.store_yaml()

        results: DetMetrics = self.yolo_model.val(
            data=config_path.as_posix(),
            split="test",
            save_json=False,
            plots=False,
            verbose=False,
            device=self.device,
            conf=self.conf,
            iou=0.45,
        )
        map50 = float(results.box.map50)

        # Ultralytics leaves circular references in memory (validator, batch,
        # predictions) that contain tensors on the GPU. Without explicit garbage
        # collection, memory grows with each evaluation, leading to a "CUDA out of memory"
        # error after a few dozen trials.
        del results
        free_gpu_memory()

        return map50

    def train_and_evaluate(self, dataset: Dataset, epochs: int = 5) -> float:
        """
        Trains a 'fresh' YOLO model to avoid data leakage between trials,
        then evaluates it on the filtered temporary dataset.
        """
        from ultralytics import YOLO
        import tempfile

        if not dataset.samples:
            raise ValueError("Cannot evaluate an empty dataset.")

        dataset_root = getattr(dataset, "base_path", None)
        if dataset_root is None:
            dataset_root = dataset.samples[0].image_path.parent.parent

        config = YoloConfig(
            path=Path(dataset_root),
            names=list(CLASS_NAMES),
            nc=4
        )
        config_path = config.store_yaml()

        # 1. Load a FRESH copy of the model to avoid data leakage
        fresh_model = YOLO(self.model_path.as_posix())

        # 2. Train and evaluate in a temporary environment to avoid cluttering the cluster with files
        with tempfile.TemporaryDirectory() as temp_dir:
            fresh_model.train(
                data=config_path.as_posix(),
                epochs=epochs,
                project=temp_dir,
                name="optuna_trial",
                device=self.device,
                verbose=False,
                plots=False,
                exist_ok=True
            )
            
            results = fresh_model.val(
                data=config_path.as_posix(),
                split="test",
                device=self.device,
                verbose=False,
                plots=False,
                save_json=False
            )

        return results.box.map50


if __name__ == "__main__":
    import tempfile
    import filters

    dataset = Dataset.from_yolo_directory(Path("resources/dataset"))
    model = Yolo.load_model(Path("resources/model.pt"))

    samples = dataset.pick_random_samples(
        sample_count=50,
        domain="target", 
        split=DatasetSplit.TEST
    )

    best_parameters: List[filters.ParametrizedFilter] = [
        filters.BilateralFilterAdapter.parametrized(
            filters.BilateralFilterParameters(
                diameter=8,
                sigmaColor=65,
                sigmaSpace=45,
            ),
        ),
        filters.SaturationBoostFilterAdapter.parametrized(
            filters.SaturationBoostParameters(boost_factor=1.8)
        ),
        filters.ClaheFilterAdapter.parametrized(
            filters.ClaheParameters(
                clip_limit=2.0,
                tile_grid_size=16,
            )
        ),
        filters.UnsharpMaskFilterAdapter.parametrized(
            filters.UnsharpMaskParameters(
                sigma=1.6,
                strength=1.75,
            )
        ),
    ]

    mini_dataset = Dataset(samples=samples)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        eval_path = Path(temp_dir) / "eval_run"
        staged_dataset = mini_dataset.apply_filters_and_export(
            filters=best_parameters,
            destination_path=eval_path
        )
        result = model.evaluate(staged_dataset)
        print(f"Test evaluation mAP@50: {result}")