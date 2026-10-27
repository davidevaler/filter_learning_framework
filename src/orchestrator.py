"""
orchestrator.py

This module provides a framework for orchestrating the optimization of image preprocessing
filter combinations using Optuna. The orchestrator implements a greedy layer-by-layer
strategy: at each layer every candidate filter is optimized on top of the filters selected
in the previous layers, whose parameters are kept FROZEN. The best candidate is then fixed
and the search moves to the next layer.

The number of trials of each study is derived automatically from the search space of the
filter under optimization (see trainer.trial_budget), and a study is stopped early when it
stops improving.
"""

import json
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
)

import optuna
from pydantic import BaseModel, Field, PlainSerializer

from .dataset import Dataset
from .filters import (
    BilateralFilterAdapter,
    ClaheFilterAdapter,
    FilterAdapter,
    GammaCorrectionFilterAdapter,
    LaplacianSharpenFilterAdapter,
    MedianBlurFilterAdapter,
    NoOpFilterAdapter,
    ParametrizedFilter,
    SaturationBoostFilterAdapter,
    UnsharpMaskFilterAdapter,
)
from .trainer import (
    Trainer,
    describe_search_space,
    deserialize_pipeline,
    trial_budget,
)
from .yolo import Yolo


class PatienceCallback:
    """
    Stops an Optuna study when the best value has not improved by at least
    `min_delta` for `patience` consecutive trials.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.stale_trials = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            return

        if self.best is None or trial.value > self.best + self.min_delta:
            self.best = trial.value
            self.stale_trials = 0
        else:
            self.stale_trials += 1

        if self.stale_trials >= self.patience:
            print(
                f"  Early stopping: no improvement > {self.min_delta} "
                f"in the last {self.patience} trials."
            )
            study.stop()


class TrialResult(BaseModel):
    trial_number: Annotated[
        int, Field(..., description="Trial number in the Optuna study")
    ]
    map_50: Annotated[
        float,
        Field(..., description="Mean Average Precision at IoU 0.5 for this trial"),
    ]
    filters: Annotated[
        List[ParametrizedFilter],
        Field(..., description="Parametrized filters used in this trial"),
        PlainSerializer(
            lambda filters: [
                {
                    "name": f.adapter.name,
                    "parameters": f.parameters.model_dump(),
                }
                for f in filters
            ],
            List[Dict[str, Any]],
        ),
    ]


class FilterOptimizationStudy(BaseModel):
    study_name: Annotated[
        str, Field(..., description="Unique name for the Optuna study")
    ]
    filters: Annotated[
        List[Type[FilterAdapter]],
        Field(..., description="Filter adapter classes in this combination"),
        PlainSerializer(
            lambda filters: [f.name for f in filters],
            List[str],
        ),
    ]
    best_trial: Annotated[
        Optional[TrialResult],
        Field(default=None, description="Best trial result found so far"),
    ] = None
    all_trials: Annotated[
        List[TrialResult],
        Field(..., description="All completed trial results"),
    ] = []
    completed_trials_count: Annotated[
        int,
        Field(default=0, description="Number of completed trials for this combination"),
    ] = 0
    trial_budget: Annotated[
        Optional[int],
        Field(default=None, description="Maximum number of trials computed for this study"),
    ] = None
    search_space_size: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Number of configurations of the optimized filter (None if continuous)",
        ),
    ] = None

    def add_trial(self, trial_result: TrialResult) -> None:
        self.all_trials.append(trial_result)
        self.completed_trials_count += 1

        if self.best_trial is None or trial_result.map_50 > self.best_trial.map_50:
            self.best_trial = trial_result


class OrchestratorLog(BaseModel):
    reports: Annotated[
        List[FilterOptimizationStudy],
        Field(default_factory=list, description="All optimization studies"),
    ]
    current_layer_index: Annotated[
        int,
        Field(
            default=0,
            description="Current layer index being processed (0-based, resumable)",
        ),
    ] = 0
    best_map_50: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Best mean Average Precision at IoU 0.5 found so far",
        ),
    ] = None
    best_filters_combination: Annotated[
        List[ParametrizedFilter],
        Field(
            default_factory=list,
            description="Best filter combination found so far",
        ),
        PlainSerializer(
            lambda filters: [
                {
                    "name": f.adapter.name,
                    "parameters": f.parameters.model_dump(),
                }
                for f in filters
            ],
            List[Dict[str, Any]],
        ),
    ]

    def save(self, log_file_path: Path) -> None:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_path.write_text(self.model_dump_json(indent=2))

    def get_report(
        self, filters: List[Type[FilterAdapter]]
    ) -> Optional[FilterOptimizationStudy]:
        filters_without_noop = [f for f in filters if f != NoOpFilterAdapter]
        filter_names = [f.name for f in filters_without_noop]

        for report in self.reports:
            report_filter_names = [
                f.name for f in report.filters if f != NoOpFilterAdapter
            ]
            if report_filter_names == filter_names:
                return report
        return None

    def add_or_get_report(
        self, filters: List[Type[FilterAdapter]], study_name: str
    ) -> FilterOptimizationStudy:
        existing_report = self.get_report(filters)
        if existing_report is not None:
            return existing_report

        new_report = FilterOptimizationStudy(
            filters=filters,
            study_name=study_name,
        )
        self.reports.append(new_report)
        return new_report

    def update_best_if_needed(self, report: FilterOptimizationStudy) -> bool:
        if report.best_trial is None:
            return False

        if self.best_map_50 is None or report.best_trial.map_50 > self.best_map_50:
            self.best_map_50 = report.best_trial.map_50
            self.best_filters_combination = report.best_trial.filters
            return True
        return False


class OrchestratorConfig(BaseModel):
    filter_layers: Annotated[
        List[List[Type[FilterAdapter]]],
        Field(
            ...,
            description=(
                "List of filter layers. Each layer is a list of filter adapter classes. "
                "The orchestrator optimizes one filter from each layer sequentially."
            ),
        ),
        PlainSerializer(
            lambda layers: [[f.name for f in layer] for layer in layers],
        ),
    ]
    n_trials_per_combination: Annotated[
        int,
        Field(
            default=100,
            ge=1,
            description=(
                "Hard upper bound on the trials of a single study. The actual number "
                "is computed automatically and never exceeds this value."
            ),
        ),
    ] = 100
    search_top_fraction: Annotated[
        float,
        Field(
            default=0.10,
            gt=0.0,
            lt=1.0,
            description=(
                "Fraction of the search space considered 'good enough'. With 0.10 the "
                "budget aims at finding a configuration in the best 10% of the space."
            ),
        ),
    ] = 0.10
    search_confidence: Annotated[
        float,
        Field(
            default=0.95,
            gt=0.0,
            lt=1.0,
            description="Probability of hitting the top fraction of the space.",
        ),
    ] = 0.95
    trials_per_parameter: Annotated[
        int,
        Field(
            default=10,
            ge=1,
            description="Minimum number of trials per searched hyperparameter.",
        ),
    ] = 10
    early_stopping_patience: Annotated[
        Optional[int],
        Field(
            default=15,
            ge=1,
            description=(
                "Stop a study after this many consecutive trials without improvement. "
                "None disables early stopping. Not applied to exhaustive grid searches."
            ),
        ),
    ] = 15
    early_stopping_min_delta: Annotated[
        float,
        Field(
            default=0.005,
            ge=0.0,
            description=(
                "Minimum mAP50 gain that counts as an improvement. Should be of the "
                "order of the noise of the metric on the validation set."
            ),
        ),
    ] = 0.005
    sampler_seed: Annotated[
        Optional[int],
        Field(default=None, description="Seed of the Optuna samplers, for reproducibility."),
    ] = None
    optuna_db_path: Annotated[
        Path,
        Field(..., description="Path to Optuna SQLite database for storing studies"),
    ]
    checkpoint_path: Annotated[
        Path,
        Field(..., description="Path to save/load orchestrator checkpoints"),
    ]
    baseline_map_50: Annotated[
        Optional[float],
        Field(
            default=None,
            description="mAP50 of the model WITHOUT any filter. ",
        ),
    ] = None
    study_name_prefix: Annotated[
        str,
        Field(
            default="",
            description="Prefix to use for Optuna study names (empty for no prefix)",
        ),
    ] = ""
    trial_callback: Annotated[
        Optional[Callable[[TrialResult, FilterOptimizationStudy], None]],
        Field(
            default=None,
            description="Optional callback function called after each trial completes.",
        ),
    ] = None
    layer_callback: Annotated[
        Optional[
            Callable[
                [
                    int,
                    List[Type[FilterAdapter]],
                    Optional[float],
                    List[ParametrizedFilter],
                ],
                None,
            ]
        ],
        Field(
            default=None,
            description="Optional callback function called after each layer completes.",
        ),
    ] = None
    optuna_study_kwargs: Annotated[
        Dict[str, Any],
        Field(
            description=(
                "Additional keyword arguments to pass to optuna.create_study. "
                "If it contains 'sampler', the automatic sampler choice is skipped."
            ),
        ),
    ] = {}

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def create_default(
        cls,
        optuna_db_path: Path,
        checkpoint_path: Path,
        n_trials_per_combination: int = 100,
    ) -> "OrchestratorConfig":
        return cls(
            filter_layers=[
                [NoOpFilterAdapter, MedianBlurFilterAdapter, BilateralFilterAdapter],
                [
                    NoOpFilterAdapter,
                    SaturationBoostFilterAdapter,
                    GammaCorrectionFilterAdapter,
                ],
                [
                    NoOpFilterAdapter,
                    ClaheFilterAdapter,
                ],
                [
                    NoOpFilterAdapter,
                    UnsharpMaskFilterAdapter,
                    LaplacianSharpenFilterAdapter,
                ],
            ],
            n_trials_per_combination=n_trials_per_combination,
            optuna_db_path=optuna_db_path,
            checkpoint_path=checkpoint_path,
        )

    def save(self, config_path: Path) -> None:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        serializable_data = self.model_dump(
            mode="json", exclude={"trial_callback", "layer_callback", "optuna_study_kwargs"}
        )
        config_path.write_text(json.dumps(serializable_data, indent=2))


class Orchestrator:
    @staticmethod
    def train(
        model: Yolo,
        dataset: Dataset,
        config: OrchestratorConfig,
    ) -> OrchestratorLog:
        log = OrchestratorLog()
        print("Starting new optimization run...")
        print(f"Total layers to process: {len(config.filter_layers)}")

        for layer_index in range(log.current_layer_index, len(config.filter_layers)):
            current_layer = config.filter_layers[layer_index]

            # Filters selected in the previous layers: their parameters stay frozen
            base_parametrized_filters = log.best_filters_combination.copy()
            base_combination = [f.adapter for f in base_parametrized_filters]

            layer_best_combination: Optional[List[Type[FilterAdapter]]] = None
            layer_best_parametrized_filters: Optional[List[ParametrizedFilter]] = None
            layer_best_map_50: Optional[float] = None

            print(f"\n{'=' * 60}")
            print(f"Processing Layer {layer_index + 1}/{len(config.filter_layers)}")
            print(f"Available filters: {', '.join([f.name for f in current_layer])}")
            if base_parametrized_filters:
                print("Frozen filters from previous layers:")
                for f in base_parametrized_filters:
                    print(f"  - {f.adapter.name}: {f.parameters.model_dump()}")
            print(f"{'=' * 60}\n")

            for filter_class in current_layer:
                trial_filters_combination: List[Type[FilterAdapter]] = (
                    base_combination + [filter_class]
                )

                if filter_class == NoOpFilterAdapter:
                    if base_combination:
                        # Not the first layer: NoOp reuses the best score already found
                        baseline_score = log.best_map_50
                    else:
                        # First layer: NoOp is equivalent to the pure baseline, without any filters.
                        # It must be passed from the outside because there is no existing
                        # log.best_map_50 to rely on.
                        baseline_score = config.baseline_map_50

                    print(f"Evaluating NoOp (reference score: {baseline_score})")

                    if baseline_score is not None and (
                        layer_best_map_50 is None or baseline_score > layer_best_map_50
                    ):
                        layer_best_map_50 = baseline_score
                        layer_best_combination = trial_filters_combination.copy()
                        layer_best_parametrized_filters = (
                            base_parametrized_filters.copy()
                        )
                    continue

                study_name = " + ".join([f.name for f in trial_filters_combination])
                if config.study_name_prefix:
                    study_name = f"{config.study_name_prefix}_{study_name}"

                print(f"\nOptimizing: {study_name}")

                report = log.add_or_get_report(trial_filters_combination, study_name)

                # Automatic budget: depends only on the filter being optimized,
                # because the filters from previous layers are frozen.
                space = describe_search_space([filter_class])
                budget = trial_budget(
                    space,
                    top_fraction=config.search_top_fraction,
                    confidence=config.search_confidence,
                    trials_per_parameter=config.trials_per_parameter,
                    hard_cap=config.n_trials_per_combination,
                )
                # If the budget covers the entire space, perform an exhaustive search
                use_grid = (
                    bool(space.grid)
                    and space.size is not None
                    and space.size <= budget
                )
                report.trial_budget = budget
                report.search_space_size = space.size

                size_label = "continuous" if space.size is None else str(space.size)
                print(
                    f"Search space: {space.n_parameters} parameter(s), "
                    f"{size_label} configuration(s) -> budget {budget} trial(s) "
                    f"({'exhaustive grid' if use_grid else 'TPE'})"
                )

                remaining_trials = budget - report.completed_trials_count

                if remaining_trials <= 0:
                    print(
                        f"Study already completed "
                        f"({report.completed_trials_count}/{budget} trials)"
                    )
                    print(
                        f"Best mAP@50: {report.best_trial.map_50 if report.best_trial else 'N/A'}"
                    )

                    if report.best_trial and (
                        layer_best_map_50 is None
                        or report.best_trial.map_50 > layer_best_map_50
                    ):
                        layer_best_map_50 = report.best_trial.map_50
                        layer_best_combination = trial_filters_combination.copy()
                        layer_best_parametrized_filters = (
                            report.best_trial.filters.copy()
                        )
                    continue

                study_kwargs: Dict[str, Any] = {
                    "study_name": study_name,
                    "storage": f"sqlite:///{config.optuna_db_path.as_posix()}",
                    "load_if_exists": True,
                    "direction": "maximize",
                    **config.optuna_study_kwargs,
                }
                if "sampler" not in study_kwargs:
                    if use_grid:
                        study_kwargs["sampler"] = optuna.samplers.GridSampler(
                            space.grid, seed=config.sampler_seed
                        )
                    else:
                        study_kwargs["sampler"] = optuna.samplers.TPESampler(
                            seed=config.sampler_seed
                        )
                study = optuna.create_study(**study_kwargs)

                trainer = Trainer(
                    model=model,
                    samples=dataset.samples,
                    filters_path=[filter_class],
                    frozen_filters=base_parametrized_filters,
                )

                def create_trial_callback(
                    captured_report: FilterOptimizationStudy,
                    captured_filters: List[Type[FilterAdapter]],
                    captured_budget: int,
                ) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
                    def trial_callback(
                        _study: optuna.Study, trial: optuna.trial.FrozenTrial
                    ) -> None:
                        if trial.state == optuna.trial.TrialState.COMPLETE:
                            serialized_pipeline = trial.user_attrs.get(
                                "filter_parameters"
                            )

                            if trial.value is None:
                                raise ValueError("Trial is missing 'value'.")
                            if serialized_pipeline is None:
                                raise ValueError(
                                    "Trial is missing 'filter_parameters' in user attributes."
                                )

                            # Complete pipeline: frozen filters + optimized filter
                            parametrized_filters = deserialize_pipeline(
                                serialized_pipeline
                            )

                            trial_result = TrialResult(
                                trial_number=trial.number,
                                map_50=trial.value,
                                filters=parametrized_filters,
                            )

                            captured_report.add_trial(trial_result)
                            was_updated = log.update_best_if_needed(captured_report)

                            nonlocal \
                                layer_best_map_50, \
                                layer_best_combination, \
                                layer_best_parametrized_filters
                            if (
                                layer_best_map_50 is None
                                or trial_result.map_50 > layer_best_map_50
                            ):
                                layer_best_map_50 = trial_result.map_50
                                layer_best_combination = captured_filters.copy()
                                layer_best_parametrized_filters = (
                                    trial_result.filters.copy()
                                )
                                print(
                                    f"  → New layer best! mAP@50: {layer_best_map_50:.4f}"
                                )

                            if was_updated:
                                print(
                                    f"  → New global best! mAP@50: {log.best_map_50:.4f}"
                                )

                            if config.trial_callback is not None:
                                config.trial_callback(trial_result, captured_report)

                            log.save(config.checkpoint_path)

                            print(
                                f"  Trial {trial.number} complete: mAP@50 = {trial.value:.4f}"
                            )
                            print(
                                f"  Progress: {captured_report.completed_trials_count}/"
                                f"{captured_budget} trials"
                            )

                    return trial_callback

                callbacks: List[
                    Callable[[optuna.Study, optuna.trial.FrozenTrial], None]
                ] = [create_trial_callback(report, trial_filters_combination, budget)]

                # Early stopping does not make sense on an exhaustive grid
                if config.early_stopping_patience is not None and not use_grid:
                    callbacks.append(
                        PatienceCallback(
                            patience=config.early_stopping_patience,
                            min_delta=config.early_stopping_min_delta,
                        )
                    )

                study.optimize(
                    trainer.objective,
                    n_trials=remaining_trials,
                    callbacks=callbacks,
                )

                print(f"\nStudy complete for {study_name}")
                print(
                    f"Trials used: {report.completed_trials_count}/{budget}"
                )
                if report.best_trial:
                    print(f"Best mAP@50: {report.best_trial.map_50:.4f}")

            if layer_best_parametrized_filters is None:
                layer_best_parametrized_filters = base_parametrized_filters
            if layer_best_map_50 is None:
                layer_best_map_50 = log.best_map_50
            if layer_best_combination is None:
                layer_best_combination = base_combination

            log.best_filters_combination = layer_best_parametrized_filters
            log.best_map_50 = layer_best_map_50
            log.current_layer_index = layer_index + 1
            log.save(config.checkpoint_path)

            if config.layer_callback is not None:
                config.layer_callback(
                    layer_index,
                    layer_best_combination,
                    layer_best_map_50,
                    log.best_filters_combination,
                )

            print(f"\nLayer {layer_index + 1} complete!")
            print(
                f"Best combination so far: {[f.name for f in layer_best_combination]}"
            )
            if log.best_map_50 is None:
                print("Best mAP@50: N/A")
            else:
                print(f"Best mAP@50: {log.best_map_50:.4f}")

        print(f"\n{'=' * 60}")
        print("Training complete!")
        print(
            f"Final best combination: {[f.adapter.name for f in log.best_filters_combination]}"
        )
        for f in log.best_filters_combination:
            print(f"  - {f.adapter.name}: {f.parameters.model_dump()}")
        if log.best_map_50 is None:
            print("Final best mAP@50: N/A")
        else:
            print(f"Final best mAP@50: {log.best_map_50:.4f}")
        print(f"{'=' * 60}\n")

        return log