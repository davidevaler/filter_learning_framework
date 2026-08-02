"""
orchestrator.py

This module provides a comprehensive framework for orchestrating the optimization of image
preprocessing filter combinations using Optuna. The orchestrator implements a layer-by-layer
optimization strategy where filters are selected and optimized sequentially, building upon the
best combination found in previous layers.
"""

import json
import math
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

from .dataset import Dataset, DatasetSplit, Magnitude, StoredDataset
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
    FilterParametersHint,
)
from .trainer import Trainer
from .yolo import Yolo


def _calculate_optimal_trials(filter_combination: List[Type[FilterAdapter]]) -> int:
    """
    Calcola il numero dinamico di iterazioni di Optuna necessarie 
    in base alla complessità dei filtri nella combinazione.
    """
    total_space_size = 1
    has_continuous_vars = False

    for filter_class in filter_combination:
        if filter_class.__name__ == "NoOpFilterAdapter":
            continue
            
        params_class = getattr(filter_class, "parameters_class", None)
        if not params_class:
            continue
            
        filter_space_size = 1

        for field_name, field_info in params_class.model_fields.items():
            type_hints = params_class.__annotations__.get(field_name)
            if not type_hints:
                continue

            for hint in getattr(type_hints, '__metadata__', []):
                if type(hint).__name__ == "FilterParametersHint":
                    if getattr(hint, 'step', None) is not None:
                        possibilities = math.floor((hint.upper_bound - hint.lower_bound) / hint.step) + 1
                        filter_space_size *= possibilities
                    else:
                        has_continuous_vars = True
                        filter_space_size *= 10

        total_space_size *= filter_space_size

    if total_space_size == 1 and not has_continuous_vars:
        return 1
        
    if has_continuous_vars:
        return min(max(total_space_size, 15), 40)
    else:
        return min(total_space_size, 30)


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
            report_filters_without_noop = [
                name for name in report.filters if name != "NoOpFilterAdapter"
            ]
            if report_filters_without_noop == filter_names:
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
            description="Legacy parameter. Max trials per combination now determined dynamically.",
        ),
    ] = 100
    optuna_db_path: Annotated[
        Path,
        Field(..., description="Path to Optuna SQLite database for storing studies"),
    ]
    checkpoint_path: Annotated[
        Path,
        Field(..., description="Path to save/load orchestrator checkpoints"),
    ]
    skip_all_noop: Annotated[
        bool,
        Field(
            default=True,
            description="If True, skip combinations where all filters are NoOp",
        ),
    ] = True
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
            description="Additional keyword arguments to pass to optuna.create_study.",
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
        serializable_data = self.model_dump(mode="json")
        config_path.write_text(json.dumps(serializable_data, indent=2))


class Orchestrator:
    @staticmethod
    def train(
        model: Yolo,
        dataset: StoredDataset,
        config: OrchestratorConfig,
    ) -> OrchestratorLog:
        log = OrchestratorLog()
        print("Starting new optimization run...")
        print(f"Total layers to process: {len(config.filter_layers)}")

        for layer_index in range(log.current_layer_index, len(config.filter_layers)):
            current_layer = config.filter_layers[layer_index]

            base_combination = [f.adapter for f in log.best_filters_combination]
            base_parametrized_filters = log.best_filters_combination.copy()

            layer_best_combination: Optional[List[Type[FilterAdapter]]] = None
            layer_best_parametrized_filters: Optional[List[ParametrizedFilter]] = None
            layer_best_map_50: Optional[float] = None

            print(f"\n{'=' * 60}")
            print(f"Processing Layer {layer_index + 1}/{len(config.filter_layers)}")
            print(f"Available filters: {', '.join([f.name for f in current_layer])}")
            print(f"{'=' * 60}\n")

            for filter_class in current_layer:
                trial_filters_combination: List[Type[FilterAdapter]] = (
                    base_combination + [filter_class]
                )

                if filter_class == NoOpFilterAdapter and base_combination:
                    print("Skipping redundant NoOp combination (using base result)")
                    if log.best_map_50 is not None and (
                        layer_best_map_50 is None or log.best_map_50 > layer_best_map_50
                    ):
                        layer_best_map_50 = log.best_map_50
                        layer_best_combination = trial_filters_combination.copy()
                        layer_best_parametrized_filters = (
                            base_parametrized_filters.copy()
                        )
                    continue

                if config.skip_all_noop and all(
                    f == NoOpFilterAdapter for f in trial_filters_combination
                ):
                    print("Skipping all-NoOp combination")
                    continue

                study_name = " + ".join([f.name for f in trial_filters_combination])
                if config.study_name_prefix:
                    study_name = f"{config.study_name_prefix}_{study_name}"

                print(f"\nOptimizing: {study_name}")

                report = log.add_or_get_report(trial_filters_combination, study_name)

                # Calcolo dinamico delle iterazioni ottimali
                optimal_trials = _calculate_optimal_trials(trial_filters_combination)
                remaining_trials = optimal_trials - report.completed_trials_count

                if remaining_trials <= 0:
                    print(
                        f"Study already completed "
                        f"({report.completed_trials_count}/{optimal_trials} trials)"
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

                print(
                    f"Resuming study: {report.completed_trials_count}/"
                    f"{optimal_trials} trials completed"
                )

                study_kwargs = {
                    "study_name": study_name,
                    "storage": f"sqlite:///{config.optuna_db_path.as_posix()}",
                    "load_if_exists": True,
                    "direction": "maximize",
                    **config.optuna_study_kwargs,
                }
                study = optuna.create_study(**study_kwargs)

                trainer = Trainer(
                    model=model,
                    samples=dataset.samples,
                    filters_path=trial_filters_combination,
                )

                def create_trial_callback(
                    captured_report: FilterOptimizationStudy,
                    captured_filters: List[Type[FilterAdapter]],
                ) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
                    def trial_callback(
                        _study: optuna.Study, trial: optuna.trial.FrozenTrial
                    ) -> None:
                        if trial.state == optuna.trial.TrialState.COMPLETE:
                            parametrized_filters_dict: Optional[Dict[str, Dict]] = (
                                trial.user_attrs.get("filter_parameters")
                            )

                            if trial.value is None:
                                raise ValueError("Trial is missing 'value'.")
                            if parametrized_filters_dict is None:
                                raise ValueError(
                                    "Trial is missing 'filter_parameters' in user attributes."
                                )

                            parametrized_filters = []
                            for (
                                filter_name,
                                parameters,
                            ) in parametrized_filters_dict.items():
                                base_name = filter_name.rsplit("_", 1)[0]
                                filter_adapter = FilterAdapter.from_name(base_name)
                                parameters_cls = getattr(
                                    filter_adapter, "parameters_class"
                                )  # type: ignore
                                parametrized_filter = ParametrizedFilter(
                                    adapter=filter_adapter,
                                    parameters=parameters_cls(**parameters),  # type: ignore
                                )
                                parametrized_filters.append(parametrized_filter)

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
                                f"{optimal_trials} trials"
                            )

                    return trial_callback

                trial_callback = create_trial_callback(
                    report, trial_filters_combination
                )

                study.optimize(
                    trainer.objective,
                    n_trials=remaining_trials,
                    callbacks=[trial_callback],
                )

                print(f"\nStudy complete for {study_name}")
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
        if log.best_map_50 is None:
            print("Final best mAP@50: N/A")
        else:
            print(f"Final best mAP@50: {log.best_map_50:.4f}")
        print(f"{'=' * 60}\n")

        return log