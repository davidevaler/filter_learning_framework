"""
dataset.py
A generalized, memory-efficient framework for handling image datasets and object detection labels.
"""

import uuid
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import Self


class DatasetSplit(str, Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"


class Label(BaseModel):
    """Generic bounding box representation (normalized coordinates)."""
    class_id: int
    top_left: float
    top_right: float
    bottom_left: float
    bottom_right: float

    @classmethod
    def from_yolo_string(cls, line: str) -> Self:
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError("Invalid YOLO format string. Expected 5 space-separated values.")
        return cls(
            class_id=int(parts[0]),
            top_left=float(parts[1]),
            top_right=float(parts[2]),
            bottom_left=float(parts[3]),
            bottom_right=float(parts[4]),
        )

    def to_yolo_string(self) -> str:
        return f"{self.class_id} {self.top_left} {self.top_right} {self.bottom_left} {self.bottom_right}"


class Sample(BaseModel):
    """
    Unified representation of a sample. 
    Handles lazy-loading to keep memory footprint minimal.
    """
    image_path: Path
    labels: List[Label] = Field(default_factory=list)
    domain: Optional[str] = None
    dataset_split: Optional[DatasetSplit] = None
    
    _cached_numpy_image: Optional[np.ndarray] = PrivateAttr(default=None)

    def load_image(self) -> np.ndarray:
        if self._cached_numpy_image is None:
            import cv2
            img = cv2.imread(str(self.image_path))
            if img is None:
                raise Exception(f"Failed to load image '{self.image_path}'.")
            self._cached_numpy_image = img
        return self._cached_numpy_image.copy()

    def unload_image(self):
        """Frees the RAM used by the image array."""
        self._cached_numpy_image = None


class Dataset:
    """
    Generalized Dataset manager. 
    Decoupled from physical file formats, works entirely in memory with lazy-loaded samples.
    """
    def __init__(self, samples: List[Sample] = None):
        self.samples: List[Sample] = samples if samples is not None else []

    # --- I/O ADAPTERS to also use later with KNIME dataframes ---

    @classmethod
    def from_yolo_directory(cls, path: Path) -> Self:
        """Builds a Dataset from a YOLO-formatted directory structure. Expects 'images' and 'labels' subdirectories."""
        from tqdm import tqdm
        
        if not path.exists() or not path.is_dir():
            raise Exception(f"Invalid directory: '{path}'.")

        samples = []
        image_extensions = {".png", ".jpg", ".jpeg"}
        
        for img_path in tqdm(path.rglob("*"), desc="Scanning YOLO directory"):
            if img_path.suffix.lower() not in image_extensions:
                continue

            # Infer label path based on YOLO conventions
            img_str = str(img_path).replace("\\", "/")
            if "/images/" in img_str:
                lbl_str = img_str.replace("/images/", "/labels/").rsplit('.', 1)[0] + '.txt'
                lbl_path = Path(lbl_str)
            else:
                lbl_path = img_path.with_suffix(".txt")

            labels = []
            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    if line.strip():
                        labels.append(Label.from_yolo_string(line))

            # Infer dataset split and domain from the directory structure
            split = next((DatasetSplit(p.lower()) for p in img_path.parts if p.lower() in [s.value for s in DatasetSplit]), None)
            rel_parts = img_path.relative_to(path).parts
            domain = rel_parts[0] if len(rel_parts) > 1 and rel_parts[0] not in ['images', 'labels'] else None

            samples.append(Sample(
                image_path=img_path,
                labels=labels,
                domain=domain,
                dataset_split=split
            ))

        return cls(samples=samples)

    # KNIME integration: placeholder for future implementation
    @classmethod
    def from_dataframe(cls, df, image_col: str, labels_col: str) -> Self:
        """To be implemented for KNIME integration."""
        pass

    # --- CORE OPERATIONS ---

    def pick_random_samples(self, sample_count: int, domain: Optional[str] = None, split: Optional[DatasetSplit] = None) -> List[Sample]:
        import random
        filtered = self.samples
        if domain:
            filtered = [s for s in filtered if s.domain == domain]
        if split:
            filtered = [s for s in filtered if s.dataset_split == split]
        
        if sample_count > len(filtered):
            raise Exception(f"Requested {sample_count}, but only {len(filtered)} available.")
        return random.sample(filtered, sample_count)

    def apply_filters_and_export(
        self, 
        filters: List["ParametrizedFilter"], 
        destination_path: Path
    ) -> Self:
        """
        Applies a series of filters to each sample and exports the results to a YOLO-formatted directory.
        """
        import cv2
        from tqdm import tqdm

        if destination_path.exists():
            raise Exception(f"Destination path '{destination_path}' already exists.")

        images_dir = destination_path / "images"
        labels_dir = destination_path / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        processed_samples = []

        for sample in tqdm(self.samples, desc="Filtering and exporting", unit="sample"):
            img_array = sample.load_image()
            
            for f in filters:
                img_array = f.apply(img_array)
                
            unique_name = uuid.uuid4().hex
            out_img = images_dir / f"{unique_name}.png"
            out_lbl = labels_dir / f"{unique_name}.txt"
            
            cv2.imwrite(str(out_img), img_array)
            out_lbl.write_text("\n".join([lbl.to_yolo_string() for lbl in sample.labels]))
            
            # Free up memory after processing each sample
            sample.unload_image()
            
            processed_samples.append(Sample(
                image_path=out_img,
                labels=sample.labels,
                domain=sample.domain,
                dataset_split=sample.dataset_split
            ))

        return type(self)(samples=processed_samples)