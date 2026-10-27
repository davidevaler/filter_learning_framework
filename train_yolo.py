"""
train_yolo.py

Trains YOLOv3u on the HCM images of the M5 dataset (1000x subset prepared by
Senette and Andreolli), with or without the targeted data augmentation based on
the filters, and finally evaluates the best model on the HCM and LCM test sets.

Based on the yolo_training.ipynb notebook of the original framework, with these
differences:
  1. explicit splits: train = images/train, val = images/val, test = images/test.
     In the notebook YoloConfig uses "images" for all three, and Ultralytics
     searches for images recursively: the model would also be trained on the
     HCM validation and test images, which show the same fields of view as the
     LCM test set;
  2. the augmentation filters are applied in BGR order, as in the pipeline
     search and in the KNIME node (Ultralytics passes the images in RGB);
  3. a custom trainer instead of monkey-patching build_yolo_dataset;
  4. fixed seed and final evaluation on HCM and LCM, with conf=0.001 (standard
     value for mAP) and conf=0.25 (value used by yolo.py in the original
     framework).

Usage, from the root folder of the framework:
    python train_yolo.py --dataset resources/dataset --da --name with_da
    python train_yolo.py --dataset resources/dataset --name no_da

Expected dataset structure (the one of the Kaggle dataset of the framework):
    <dataset>/source/images/{train,val,test}/*.png    HCM
    <dataset>/source/labels/{train,val,test}/*.txt
    <dataset>/target/images/{train,val,test}/*.png    LCM
    <dataset>/target/labels/{train,val,test}/*.txt
"""

import argparse
import importlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np

# Order of the IDs as in the COCO files of the dataset (resources/dataset/coco_annotations)
CLASS_NAMES = ["schizont", "gametocyte", "ring", "trophozoite"]

# One random filter for each layer, as in the notebook of the original framework.
# Note: these layers group the filters differently from the pipeline search
# (gamma correction is together with CLAHE here), as in the original notebook.
FILTER_LAYERS = [
    ["NoOpFilterAdapter", "MedianBlurFilterAdapter", "BilateralFilterAdapter"],
    ["NoOpFilterAdapter", "SaturationBoostFilterAdapter"],
    ["NoOpFilterAdapter", "ClaheFilterAdapter", "GammaCorrectionFilterAdapter"],
    ["NoOpFilterAdapter", "UnsharpMaskFilterAdapter", "LaplacianSharpenFilterAdapter"],
]


# ---------------------------------------------------------------------------
# Targeted data augmentation (numpy only, can be tested without torch)
# ---------------------------------------------------------------------------


def load_filters(package):
    """Return the layers of FILTER_LAYERS with the classes of the <package>.filters module."""
    module = importlib.import_module(f"{package}.filters")
    return [[getattr(module, name) for name in layer] for layer in FILTER_LAYERS], module


def random_parameters(filter_class, hint_class, rng):
    """Random values, within the bounds declared in the framework, for the parameters of a filter."""
    params_class = filter_class.parameters_class
    values = {}
    for field_name, field_info in params_class.model_fields.items():
        annotated = params_class.__annotations__.get(field_name)
        hint = next(
            (h for h in getattr(annotated, "__metadata__", ()) if isinstance(h, hint_class)),
            None,
        )
        if hint is None:
            continue
        low, high, step = hint.lower_bound, hint.upper_bound, hint.step
        if low == high:
            values[field_name] = field_info.annotation(low)
        elif field_info.annotation is int:
            values[field_name] = rng.choice(range(int(low), int(high) + 1, int(step or 1)))
        elif step is None:
            values[field_name] = rng.uniform(low, high)
        else:
            # the epsilon avoids losing the last value because of rounding errors
            count = math.floor((high - low) / step + 1e-9) + 1
            values[field_name] = round(low + rng.randrange(count) * step, 10)
    return params_class(**values)


def augment_bgr(image_bgr, layers, hint_class, rng=random):
    """Apply one random filter per layer to a BGR uint8 image (H, W, 3)."""
    image = np.ascontiguousarray(image_bgr)
    for layer in layers:
        filter_class = rng.choice(layer)
        image = filter_class.apply_filter(image, random_parameters(filter_class, hint_class, rng))
        image = np.ascontiguousarray(image, dtype=np.uint8)
    return image


# ---------------------------------------------------------------------------
# Integration with Ultralytics
# ---------------------------------------------------------------------------


def make_trainer_class(layers, hint_class):
    import torch
    from ultralytics.data import YOLODataset
    from ultralytics.models.yolo.detect import DetectionTrainer

    class FilterAugmentedDataset(YOLODataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            # Ultralytics returns a uint8 tensor (C, H, W) in RGB order,
            # already resized to the training size (imgsz)
            rgb = sample["img"].permute(1, 2, 0).cpu().numpy()
            bgr = augment_bgr(rgb[..., ::-1], layers, hint_class)
            rgb = np.ascontiguousarray(bgr[..., ::-1])
            sample["img"] = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
            return sample

    class FilterAugmentedTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            if mode == "train":
                # augmentation only on the training images, not on validation
                dataset.__class__ = FilterAugmentedDataset
            return dataset

    return FilterAugmentedTrainer


def write_data_yaml(domain_root, out_path):
    lines = [
        f"path: {domain_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ] + [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def check_dataset(root):
    counts = {}
    for domain in ("source", "target"):
        for split in ("train", "val", "test"):
            folder = root / domain / "images" / split
            if not folder.is_dir():
                raise SystemExit(f"Missing folder: {folder}")
            counts[f"{domain}/{split}"] = len(list(folder.glob("*.png")))
    print("Images found:", counts)
    return counts


def evaluate(weights, yaml_path, device, conf):
    from ultralytics import YOLO

    r = YOLO(str(weights)).val(
        data=str(yaml_path), split="test", device=device, conf=conf, iou=0.45,
        plots=False, save_json=False, verbose=False,
    )
    return {
        "mAP50": float(r.box.map50),
        "mAP50_95": float(r.box.map),
        "precision": float(r.box.mp),
        "recall": float(r.box.mr),
        "mAP50_per_class": {
            CLASS_NAMES[int(c)]: float(v) for c, v in zip(r.box.ap_class_index, r.box.ap50)
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--da", action="store_true", help="enable the targeted data augmentation")
    p.add_argument("--name", required=True, help="name of the experiment (results folder)")
    p.add_argument("--weights", default="yolov3u.pt", help="pre-trained starting weights")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", type=Path, default=Path("runs"))
    p.add_argument("--package", default="src", help="Python package of the framework")
    args = p.parse_args()

    check_dataset(args.dataset)
    out_dir = args.project / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    hcm_yaml = write_data_yaml(args.dataset / "source", out_dir / "data_hcm.yaml")
    lcm_yaml = write_data_yaml(args.dataset / "target", out_dir / "data_lcm.yaml")

    from ultralytics import YOLO

    trainer = None
    if args.da:
        layers, module = load_filters(args.package)
        trainer = make_trainer_class(layers, module.FilterParametersHint)
        print("Targeted data augmentation enabled")

    model = YOLO(args.weights)
    start = time.perf_counter()
    results = model.train(
        trainer=trainer,
        data=str(hcm_yaml),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
        optimizer="SGD", lr0=0.01, momentum=0.9, patience=100,
        device=args.device, seed=args.seed, deterministic=True,
        project=str(args.project.resolve()), name=args.name, exist_ok=True,
        pretrained=True, plots=False,
        # standard Ultralytics augmentations disabled, as in the original framework
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0,
        shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0,
        copy_paste=0.0,
    )

    train_seconds = time.perf_counter() - start

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best model: {best}")
    print(f"Training time: {train_seconds / 60:.1f} minutes")

    import torch
    import ultralytics

    summary = {
        "name": args.name,
        "data_augmentation": args.da,
        "weights": str(best),
        "seed": args.seed,
        "epochs": args.epochs,
        "train_minutes": round(train_seconds / 60, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }
    for conf in (0.001, 0.25):
        for domain, yaml_path in (("HCM", hcm_yaml), ("LCM", lcm_yaml)):
            m = evaluate(best, yaml_path, args.device, conf)
            summary[f"{domain}_test_conf{conf}"] = m
            print(f"{domain} test, conf={conf}: mAP50 = {m['mAP50']:.4f}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Results saved in {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
