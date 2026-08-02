"""
dataset.py
This module defines a structured framework for handling datasets of images and their associated labels,
using Pydantic for parameter validation and type safety.

Main Components:
- Label: Represents the bounding box coordinates and generic class ID in YOLO format.
- Sample: Encapsulates an image and its labels.
- StoredSample: Sample stored on filesystem with lazy image loading.
- TransientSample: Sample with image data in memory.
- Dataset: Abstract base class for managing collections of samples.
- StoredDataset: Dataset stored on filesystem.
- StagingDataset: Temporary dataset for staging samples before persistence.
- TransientDataset: In-memory dataset for volatile data operations.
"""

import tempfile
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import Generic, List, Optional, Self, TypeVar
from typing_extensions import deprecated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class DatasetSplit(StrEnum):
    """
    Enum for dataset splits.
    """
    TRAIN = "train"
    TEST = "test"
    VAL = "val"


class Label(BaseModel):
    """
    Represents a label for an image in YOLO format.
    """
    class_id: int = Field(
        ..., description="Generic integer ID associated with the object class."
    )
    top_left: float = Field(
        ..., description="Normalized x-coordinate of bounding box center."
    )
    top_right: float = Field(
        ..., description="Normalized y-coordinate of bounding box center."
    )
    bottom_left: float = Field(..., description="Normalized width of bounding box.")
    bottom_right: float = Field(..., description="Normalized height of bounding box.")

    @classmethod
    def from_string_line(cls, line: str) -> Self:
        format_parts = line.strip().split()
        if len(format_parts) != 5:
            raise Exception(
                "Invalid YOLO format string. Expected 5 space-separated values."
            )

        return cls(
            class_id=int(format_parts[0]),
            top_left=float(format_parts[1]),
            top_right=float(format_parts[2]),
            bottom_left=float(format_parts[3]),
            bottom_right=float(format_parts[4]),
        )

    def to_string_line(self) -> str:
        return f"{self.class_id} {self.top_left} {self.top_right} {self.bottom_left} {self.bottom_right}"


class Sample(ABC, BaseModel):
    """
    Abstract base class representing a single dataset sample.
    """
    labels: List[Label] = Field(description="List of labels for the image.")

    @abstractmethod
    def load_image(self) -> np.ndarray:
        ...

    def apply_transform(self, filters: List["ParametrizedFilter"]) -> "TransientSample":
        image = self.load_image()
        for filter in filters:
            image = filter.apply(image)

        return TransientSample(
            numpy_image=image,
            labels=self.labels,
        )


class StoredSample(Sample):
    """
    Represents a sample stored on the filesystem with lazy image loading.
    """
    image_path: Path = Field(..., description="Path to the image file.")
    labels_path: Path = Field(..., description="Path to the labels file.")
    domain: Optional[str] = Field(
        default=None, description="Generic sample domain (e.g., 'source', 'target', 'clear', 'blurry')."
    )
    dataset_split: Optional[DatasetSplit] = Field(
        default=None, description="Dataset split (train, test, val)."
    )
    _cached_numpy_image: Optional[np.ndarray] = PrivateAttr(default=None)

    def load_image(self) -> np.ndarray:
        if self._cached_numpy_image is None:
            import cv2
            loaded_image = cv2.imread(str(self.image_path))
            if loaded_image is None:
                raise Exception(f"Failed to load image '{self.image_path}'.")
            self._cached_numpy_image = loaded_image
        return self._cached_numpy_image.copy()

    def unload_image(self):
        self._cached_numpy_image = None

    def to_transient_sample(
        self, unload_after_conversion: bool = True
    ) -> "TransientSample":
        loaded_image = self.load_image()
        transient_sample = TransientSample(
            labels=self.labels,
            numpy_image=loaded_image,
        )
        if unload_after_conversion:
            self.unload_image()
        return transient_sample


class TransientSample(Sample):
    """
    Represents a sample with image data held in memory (volatile/non-persistent).
    """
    numpy_image: np.ndarray = Field(...)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_image(self) -> np.ndarray:
        return self.numpy_image.copy()

    def store_image(self, image_path: Path, labels_path: Path):
        import cv2

        if image_path.exists() or image_path.is_dir():
            raise Exception(f"Invalid image path: '{image_path}'.")
        if labels_path.exists() or labels_path.is_dir():
            raise Exception(f"Invalid labels path: '{labels_path}'.")

        cv2.imwrite(str(image_path), self.numpy_image)
        labels_yolo = "\n".join([label.to_string_line() for label in self.labels])
        labels_path.write_text(labels_yolo)


SampleT = TypeVar("SampleT", bound=Sample)


class Dataset(ABC, Generic[SampleT]):
    base_path: Optional[Path] = None
    samples: List[SampleT] = []

    def __init__(self):
        self.base_path: Optional[Path] = None
        self.samples: List[SampleT] = []

    @staticmethod
    def load_from_directory(path: Path) -> "StoredDataset":
        if not path.exists() or not path.is_dir():
            raise Exception(f"Given path '{path}' is not a valid directory.")

        instance = object.__new__(StoredDataset)
        instance.base_path = path
        instance.samples = []
        instance._load_samples()
        return instance

    @staticmethod
    def create_staging_dataset(samples: List[StoredSample]) -> "StagingDataset":
        from tqdm import tqdm
        from shutil import copy2

        staging_temporary_directory = tempfile.TemporaryDirectory()
        staging_directory_path = Path(staging_temporary_directory.name)

        staging_images_directory = staging_directory_path / "images"
        staging_labels_directory = staging_directory_path / "labels"

        staging_images_directory.mkdir(parents=True, exist_ok=True)
        staging_labels_directory.mkdir(parents=True, exist_ok=True)

        staging_samples: List[StoredSample] = []

        for source_sample in tqdm(
            samples, desc="Copying samples to staging directory", unit="sample"
        ):
            if source_sample.labels_path is None:
                raise Exception("Cannot copy sample: labels_path is None.")

            staged_image_path = staging_images_directory / source_sample.image_path.name
            staged_labels_path = staging_labels_directory / source_sample.labels_path.name

            copy2(source_sample.image_path, staged_image_path)
            copy2(source_sample.labels_path, staged_labels_path)

            staging_samples.append(
                StoredSample(
                    image_path=staged_image_path,
                    labels_path=staged_labels_path,
                    labels=source_sample.labels,
                    domain=source_sample.domain,
                    dataset_split=source_sample.dataset_split,
                )
            )

        instance = object.__new__(StagingDataset)
        instance.base_path = staging_directory_path
        instance.temporary_directory = staging_temporary_directory
        instance.samples = staging_samples
        return instance

    @staticmethod
    def create_transient_dataset(samples: List[TransientSample]) -> "TransientDataset":
        instance = object.__new__(TransientDataset)
        instance.base_path = None
        instance.samples = samples
        return instance

    def unload_cached_images(self):
        from tqdm import tqdm
        for sample in tqdm(self.samples, desc="Unloading cached images", unit="sample"):
            if isinstance(sample, StoredSample):
                sample.unload_image()

    def pick_random_samples(
        self,
        sample_count: Optional[int] = None,
        domain: Optional[str] = None,
        split: Optional[DatasetSplit] = None,
    ) -> List[SampleT]:
        if sample_count is None and domain is None and split is None:
            raise Exception("Provide at least one filtering parameter.")

        import random
        filtered_samples = self.samples.copy()

        if domain is not None:
            filtered_samples = [
                s for s in filtered_samples 
                if isinstance(s, StoredSample) and s.domain == domain
            ]

        if split is not None:
            filtered_samples = [
                s for s in filtered_samples 
                if isinstance(s, StoredSample) and s.dataset_split == split
            ]

        if sample_count is not None:
            if sample_count > len(filtered_samples):
                raise Exception(
                    f"Requested {sample_count} samples but only {len(filtered_samples)} are available."
                )
            filtered_samples = random.sample(filtered_samples, sample_count)

        return filtered_samples


class StoredDataset(Dataset[StoredSample]):
    @deprecated(
        "Use 'Dataset.load_from_directory' to instantiate StoredDataset."
    )
    def __init__(self):
        raise RuntimeError("Use 'Dataset.load_from_directory' to instantiate StoredDataset.")

    def _load_samples(self):
        from tqdm import tqdm

        if self.base_path is None:
            raise Exception("Cannot load samples: base_path is not set.")

        image_extensions = {".png", ".jpg", ".jpeg"}
        image_paths_generator = (p for p in self.base_path.rglob("*") if p.suffix.lower() in image_extensions)

        for image_path_absolute in tqdm(
            image_paths_generator, desc="Loading samples from disk", unit="sample"
        ):
            # Dynamic Label resolution (YOLO standard format)
            # Replaces the last '/images/' in the path with '/labels/' and changes extension to .txt
            image_str = str(image_path_absolute)
            if "/images/" in image_str.replace("\\", "/"):
                label_str = image_str.replace("\\", "/").replace("/images/", "/labels/").rsplit('.', 1)[0] + '.txt'
                labels_file_path = Path(label_str)
            else:
                # Fallback if structure is flat
                labels_file_path = image_path_absolute.with_suffix(".txt")
            
            if not labels_file_path.exists():
                continue # Skip images without a label file

            # Try to infer Dataset Split from path
            dataset_split = None
            for part in image_path_absolute.parts:
                if part.lower() in [s.value for s in DatasetSplit]:
                    dataset_split = DatasetSplit(part.lower())
                    break
            
            # Try to infer a Domain (e.g., 'source' or 'target' from the M5 dataset structure)
            # If the dataset has a root folder like 'dataset/source/images/train/1.png', 'source' is the domain
            relative_parts = image_path_absolute.relative_to(self.base_path).parts
            domain = relative_parts[0] if len(relative_parts) > 1 and relative_parts[0] not in ['images', 'labels'] else None

            # Parse labels
            labels_file_content = labels_file_path.read_text()
            parsed_labels: List[Label] = []
            for label_line in labels_file_content.splitlines():
                if label_line.strip():
                    parsed_labels.append(Label.from_string_line(label_line))

            self.samples.append(
                StoredSample(
                    image_path=image_path_absolute,
                    labels_path=labels_file_path,
                    labels=parsed_labels,
                    domain=domain,
                    dataset_split=dataset_split,
                )
            )

    def store(self, destination_path: Path):
        if self.base_path is None:
            raise Exception("Cannot store dataset: base_path is not set.")
        if destination_path.exists():
            raise Exception(f"Destination path '{destination_path}' already exists.")

        from shutil import copytree
        destination_path.mkdir(parents=True)
        copytree(str(self.base_path), str(destination_path))


class StagingDataset(StoredDataset):
    temporary_directory: tempfile.TemporaryDirectory

    @deprecated("Use 'Dataset.create_staging_dataset' to instantiate StagingDataset.")
    def __init__(self):
        raise RuntimeError("Use 'Dataset.create_staging_dataset' to instantiate.")

    def __del__(self):
        if hasattr(self, "temporary_directory") and self.temporary_directory is not None:
            self.temporary_directory.cleanup()


class TransientDataset(Dataset[TransientSample]):
    @deprecated("Use 'Dataset.create_transient_dataset' to instantiate TransientDataset.")
    def __init__(self):
        raise RuntimeError("Use 'Dataset.create_transient_dataset' to instantiate.")

    def to_staging_dataset(self) -> StagingDataset:
        import uuid
        import tqdm

        stored_samples: List[StoredSample] = []
        staging_temporary_directory = tempfile.TemporaryDirectory()
        staging_directory_path = Path(staging_temporary_directory.name)

        for transient_sample in tqdm.tqdm(
            self.samples, desc="Storing samples in staging directory", unit="sample"
        ):
            unique_sample_name = uuid.uuid4().hex
            staged_image_path = staging_directory_path / "images" / f"{unique_sample_name}.png"
            staged_image_path.parent.mkdir(parents=True, exist_ok=True)

            staged_labels_path = staging_directory_path / "labels" / f"{unique_sample_name}.txt"
            staged_labels_path.parent.mkdir(parents=True, exist_ok=True)

            transient_sample.store_image(image_path=staged_image_path, labels_path=staged_labels_path)

            stored_samples.append(
                StoredSample(
                    image_path=staged_image_path,
                    labels_path=staged_labels_path,
                    labels=transient_sample.labels,
                )
            )

        instance = object.__new__(StagingDataset)
        instance.base_path = staging_directory_path
        instance.temporary_directory = staging_temporary_directory
        instance.samples = stored_samples
        return instance

    def store(self, destination_path: Path) -> StoredDataset:
        import uuid
        import cv2
        from tqdm import tqdm

        if destination_path.exists() or destination_path.is_file():
            raise Exception(f"Invalid destination path '{destination_path}'.")

        destination_path.mkdir(parents=True, exist_ok=True)
        destination_images_directory = destination_path / "images"
        destination_labels_directory = destination_path / "labels"

        destination_images_directory.mkdir(parents=True, exist_ok=True)
        destination_labels_directory.mkdir(parents=True, exist_ok=True)

        persisted_samples: List[StoredSample] = []

        for transient_sample in tqdm(self.samples, desc="Persisting samples to disk", unit="sample"):
            unique_sample_name = uuid.uuid4().hex
            persisted_image_path = destination_images_directory / f"{unique_sample_name}.png"
            persisted_labels_path = destination_labels_directory / f"{unique_sample_name}.txt"

            cv2.imwrite(str(persisted_image_path), transient_sample.numpy_image)
            labels_yolo_format = "\n".join([label.to_string_line() for label in transient_sample.labels])
            persisted_labels_path.write_text(labels_yolo_format)

            persisted_samples.append(
                StoredSample(
                    image_path=persisted_image_path,
                    labels_path=persisted_labels_path,
                    labels=transient_sample.labels,
                )
            )

        instance = object.__new__(StoredDataset)
        instance.base_path = destination_path
        instance.samples = persisted_samples
        return instance