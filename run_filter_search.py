"""
run_filter_search.py

Greedy search of the filter pipeline (framework orchestrator) for an already
trained model, followed by the evaluation of the pipeline found on the LCM
test set.

Steps:
  1. mAP50 of the model without filters on the images used for the search
     (reference value of the search);
  2. pipeline search, with an automatic number of trials, on the LCM
     validation images (default) or on a subset of the LCM training images;
  3. evaluation on the LCM test set, without filters and with the pipeline
     found, with conf=0.001 (standard mAP) and conf=0.25 (original value of
     the framework).

All the results are saved in <out>/<name>/:
  optuna.db          Optuna studies
  config.json        search configuration
  checkpoint.json    orchestrator log (can also be read by the KNIME node
                     Malaria Preprocessing, "From file" option)
  pipeline.json      the pipeline found, as a list of filters
  result.json        summary: reference value, final mAP, trials used, times

Usage, from the root folder of the framework:
    # search on the 45 LCM validation images
    python run_filter_search.py --model runs/with_da_s0/weights/best.pt --name with_da_s0
    # search on 200 random LCM training images
    python run_filter_search.py --model runs/with_da_s0/weights/best.pt \
        --name with_da_s0_train200 --search-split train --search-n 200
"""

import argparse
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

from src.dataset import Dataset, DatasetSplit
from src.orchestrator import Orchestrator, OrchestratorConfig
from src.yolo import CLASS_NAMES, Yolo, YoloConfig, free_gpu_memory


def split_samples(dataset, domain, split):
    samples = [s for s in dataset.samples if s.domain == domain and s.dataset_split == split]
    if not samples:
        raise SystemExit(f"No images for domain '{domain}' and split '{split.value}'.")
    return samples


def staged(samples, filters, tmp):
    return Dataset(samples=samples).apply_filters_and_export(
        filters=filters, destination_path=Path(tmp) / "staged"
    )


def full_metrics(model, staged_dataset, conf):
    """Overall and per-class mAP50 on a dataset that has already been filtered and exported."""
    root = staged_dataset.samples[0].image_path.parent.parent
    cfg = YoloConfig(path=root, names=list(CLASS_NAMES), nc=len(CLASS_NAMES)).store_yaml()
    r = model.yolo_model.val(
        data=cfg.as_posix(), split="test", device=model.device, conf=conf, iou=0.45,
        plots=False, save_json=False, verbose=False,
    )
    metrics = {
        "mAP50": float(r.box.map50),
        "mAP50_95": float(r.box.map),
        "precision": float(r.box.mp),
        "recall": float(r.box.mr),
        "mAP50_per_class": {
            CLASS_NAMES[int(c)]: float(v) for c, v in zip(r.box.ap_class_index, r.box.ap50)
        },
    }
    del r
    free_gpu_memory()
    return metrics


def gpu_memory_callback(trial_result, report):
    """Print the GPU memory after each trial, to check that it does not grow."""
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 2**20:.0f} MiB")
    except ImportError:
        pass


def serialize(filters):
    return [{"name": f.adapter.name, "parameters": f.parameters.model_dump()} for f in filters]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--dataset", type=Path, default=Path("resources/dataset"))
    p.add_argument("--domain", default="target", help="LCM domain (folder name)")
    p.add_argument("--out", type=Path, default=Path("runs/search"))
    p.add_argument("--conf", type=float, default=0.001, help="confidence threshold used during the search")
    p.add_argument("--seed", type=int, default=0, help="seed of the Optuna samplers")
    p.add_argument("--device", default="cuda")
    p.add_argument("--search-split", choices=["val", "train"], default="val",
                   help="LCM images used for the search: validation (45) or training (837)")
    p.add_argument("--search-n", type=int, default=None,
                   help="number of images used for the search (chosen at random, with --seed)")
    p.add_argument("--fresh", action="store_true", help="delete a previous search with the same name")
    args = p.parse_args()

    out_dir = args.out / args.name
    if out_dir.exists():
        if not args.fresh:
            raise SystemExit(
                f"'{out_dir}' already exists. Use --fresh to start from scratch: the orchestrator "
                "does not resume an interrupted search and would mix the trials in the database."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    model = Yolo.load_model(args.model, device=args.device, conf=args.conf)
    print("Model classes:", model.yolo_model.names)

    dataset = Dataset.from_yolo_directory(args.dataset)
    split = DatasetSplit.VAL if args.search_split == "val" else DatasetSplit.TRAIN
    val = split_samples(dataset, args.domain, split)
    if args.search_n is not None and args.search_n < len(val):
        val = random.Random(args.seed).sample(sorted(val, key=lambda s: str(s.image_path)), args.search_n)
    test = split_samples(dataset, args.domain, DatasetSplit.TEST)
    print(f"Images for the search (LCM {split.value}): {len(val)}, LCM test: {len(test)} images")

    # 1. Reference value without filters on the search images
    with tempfile.TemporaryDirectory() as tmp:
        baseline_val = model.evaluate(staged(val, [], tmp))
    print(f"LCM {split.value} mAP50 without filters: {baseline_val:.4f}")

    # 2. Search
    config = OrchestratorConfig.create_default(
        optuna_db_path=out_dir / "optuna.db",
        checkpoint_path=out_dir / "checkpoint.json",
    )
    config.baseline_map_50 = baseline_val
    config.study_name_prefix = args.name
    config.sampler_seed = args.seed
    config.trial_callback = gpu_memory_callback
    config.save(out_dir / "config.json")

    start = time.perf_counter()
    log = Orchestrator.train(model=model, dataset=Dataset(samples=val), config=config)
    search_minutes = (time.perf_counter() - start) / 60
    pipeline = log.best_filters_combination
    (out_dir / "pipeline.json").write_text(json.dumps(serialize(pipeline), indent=2))
    print(f"Search completed in {search_minutes:.1f} minutes")

    # 3. Evaluation on the LCM test set
    test_results = {}
    for conf in (0.001, 0.25):
        for label, filters in (("no_filters", []), ("pipeline", pipeline)):
            with tempfile.TemporaryDirectory() as tmp:
                m = full_metrics(model, staged(test, filters, tmp), conf)
            test_results[f"LCM_test_{label}_conf{conf}"] = m
            print(f"LCM test {label:10s} conf={conf}: mAP50 = {m['mAP50']:.4f}")

    studies = [
        {
            "study": r.study_name,
            "space_size": r.search_space_size,
            "budget": r.trial_budget,
            "trials_used": r.completed_trials_count,
            "best_val_mAP50": r.best_trial.map_50 if r.best_trial else None,
        }
        for r in log.reports
    ]
    result = {
        "name": args.name,
        "model": str(args.model),
        "search_conf": args.conf,
        "sampler_seed": args.seed,
        "search_split": split.value,
        "search_images": len(val),
        "LCM_val_no_filters": baseline_val,
        "LCM_val_best": log.best_map_50,
        "pipeline": serialize(pipeline),
        "search_minutes": round(search_minutes, 2),
        "total_trials": sum(s["trials_used"] for s in studies),
        "studies": studies,
        **test_results,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"Results saved in {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
