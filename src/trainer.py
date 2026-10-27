"""
trainer.py
This module provides a Trainer class for optimizing sequences of image preprocessing filters
using Optuna for hyperparameter search. It leverages Pydantic for parameter validation and
type safety, and supports flexible filter pipelines. The Trainer applies a series of filters
to image samples, evaluates the model on them, and integrates with Optuna to find optimal
hyperparameters.

Main Components:
- Search-space helpers: describe the hyperparameter space of a set of filters and derive
  the number of trials it deserves (trial_budget).
- Trainer: optimizes the hyperparameters of the filters in `filters_path`, keeping the
  filters selected in previous layers (`frozen_filters`) fixed.
"""

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

import optuna
from optuna import Trial

from .dataset import (
    Dataset,
    DatasetSplit,
    Sample,
)
from .filters import (
    FilterAdapter,
    FilterParametersHint,
    FilterType,
    NoOpFilterAdapter,
    ParametrizedFilter,
)
from .yolo import Yolo

# ---------------------------------------------------------------------------
# Search-space helpers
# ---------------------------------------------------------------------------


def iter_parameter_hints(
    filter_class: Type[FilterAdapter],
) -> Iterator[Tuple[str, type, FilterParametersHint]]:
    """
    Yields (field_name, field_type, hint) for every hyperparameter of a filter
    that carries a FilterParametersHint.
    """
    params_class = filter_class.parameters_class  # type: ignore[attr-defined]
    for field_name, field_info in params_class.model_fields.items():
        annotated = params_class.__annotations__.get(field_name)
        for hint in getattr(annotated, "__metadata__", ()):
            # Compared by name so it keeps working if the module gets imported
            # under two different paths (e.g. 'filters' and 'package.filters').
            if type(hint).__name__ == "FilterParametersHint":
                yield field_name, field_info.annotation, hint
                break


def is_constant(hint: FilterParametersHint) -> bool:
    """A parameter whose bounds coincide is a constant, not a search dimension."""
    return hint.lower_bound == hint.upper_bound


def grid_values(field_type: type, hint: FilterParametersHint) -> Optional[List[Any]]:
    """
    Returns the list of values a parameter can take, or None if it is continuous
    (float without step).
    """
    if is_constant(hint):
        return [field_type(hint.lower_bound)]

    if field_type is int:
        step = int(hint.step) if hint.step is not None else 1
        return list(range(int(hint.lower_bound), int(hint.upper_bound) + 1, step))

    if hint.step is None:
        return None

    # The epsilon guards against float errors: (1.4 - 0.8) / 0.1 = 5.999999999999998
    count = math.floor((hint.upper_bound - hint.lower_bound) / hint.step + 1e-9) + 1
    return [round(hint.lower_bound + i * hint.step, 10) for i in range(count)]


def optuna_param_name(filter_class: Type[FilterAdapter], field_name: str) -> str:
    return f"{filter_class.name}_{field_name}"


@dataclass
class SearchSpace:
    """
    Description of the hyperparameter space of a set of filters.

    Attributes:
        n_parameters: Number of parameters actually searched (constants excluded).
        size: Number of distinct configurations, or None if any parameter is continuous.
        grid: Optuna parameter name -> list of values, or None if any parameter is continuous.
    """

    n_parameters: int
    size: Optional[int]
    grid: Optional[Dict[str, List[Any]]]


def describe_search_space(filters: List[Type[FilterAdapter]]) -> SearchSpace:
    grid: Dict[str, List[Any]] = {}
    n_parameters = 0
    has_continuous = False

    for filter_class in filters:
        for field_name, field_type, hint in iter_parameter_hints(filter_class):
            values = grid_values(field_type, hint)
            if values is None:
                has_continuous = True
                n_parameters += 1
            elif len(values) > 1:
                grid[optuna_param_name(filter_class, field_name)] = values
                n_parameters += 1

    if has_continuous:
        return SearchSpace(n_parameters=n_parameters, size=None, grid=None)

    size = math.prod(len(values) for values in grid.values())
    return SearchSpace(n_parameters=n_parameters, size=size, grid=grid)


def trial_budget(
    space: SearchSpace,
    top_fraction: float = 0.10,
    confidence: float = 0.95,
    trials_per_parameter: int = 10,
    hard_cap: Optional[int] = None,
) -> int:
    """
    Number of trials to spend on a search space.

    The budget does not grow with the size of the grid. It is the larger of:
      - the number of random draws needed to hit the best `top_fraction` of the space
        with probability `confidence`: n = ln(1 - confidence) / ln(1 - top_fraction),
        which is independent of how many configurations exist (29 for 10% / 95%);
      - `trials_per_parameter` trials for each searched parameter.
    It is then capped by the size of the space (no point in exceeding an exhaustive
    search) and by `hard_cap`.
    """
    if space.n_parameters == 0:
        return 1

    n_random = math.ceil(math.log(1.0 - confidence) / math.log(1.0 - top_fraction))
    budget = max(n_random, trials_per_parameter * space.n_parameters)

    if space.size is not None:
        budget = min(budget, space.size)
    if hard_cap is not None:
        budget = min(budget, hard_cap)

    return max(budget, 1)


# ---------------------------------------------------------------------------
# Pipeline (de)serialization for Optuna user attributes
# ---------------------------------------------------------------------------


def serialize_pipeline(pipeline: List[ParametrizedFilter]) -> List[Dict[str, Any]]:
    return [
        {"name": f.adapter.name, "parameters": f.parameters.model_dump()}
        for f in pipeline
    ]


def deserialize_pipeline(data: List[Dict[str, Any]]) -> List[ParametrizedFilter]:
    pipeline: List[ParametrizedFilter] = []
    for entry in data:
        adapter = FilterAdapter.from_name(entry["name"])
        parameters_cls = adapter.parameters_class  # type: ignore[attr-defined]
        pipeline.append(
            ParametrizedFilter(
                adapter=adapter,
                parameters=parameters_cls(**entry["parameters"]),
            )
        )
    return pipeline


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    Optimizes the hyperparameters of the filters in `filters_path` with Optuna.

    The filters in `frozen_filters` (those selected in previous layers) are applied
    first, with their parameters fixed: Optuna only searches the new filters.
    """

    model: Yolo
    filters_path: List[Type[FilterAdapter]]
    frozen_filters: List[ParametrizedFilter]
    samples: List[Sample]

    def __init__(
        self,
        model: Yolo,
        filters_path: List[Type[FilterAdapter]],
        samples: List[Sample],
        frozen_filters: Optional[List[ParametrizedFilter]] = None,
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            model: Model wrapper exposing `evaluate(dataset) -> float`.
            filters_path: Filter adapter classes to optimize, in the order to be applied.
            samples: List of samples to process.
            frozen_filters: Already parametrized filters applied before `filters_path`.
                Their parameters are not searched.
        """
        self.frozen_filters = list(frozen_filters or [])

        # Ensure filters are ordered correctly (frozen ones included) while
        # allowing skipped layers (NoOp)
        full_path = [f.adapter for f in self.frozen_filters] + list(filters_path)
        last_filter_type_value = 0
        for i, filter_class in enumerate(full_path):
            if filter_class.filter_type == FilterType.NO_OP:
                continue
            if filter_class.filter_type <= last_filter_type_value:
                raise ValueError(
                    f"Violation of ordering constraint at layer {i}: found '{filter_class.filter_type.name}'."
                )
            last_filter_type_value = filter_class.filter_type

        # Ensure there is something to optimize
        if all(filter_class == NoOpFilterAdapter for filter_class in filters_path):
            raise ValueError("The filter path cannot consist solely of NO_OP filters.")

        self.model = model
        self.filters_path = list(filters_path)
        self.samples = samples

    def objective(self, trial: Trial) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object.

        Returns:
            mAP@50 of the model on the filtered samples.
        """
        optimized_filters: List[ParametrizedFilter] = []

        # Suggest hyperparameters only for the filters being optimized
        for filter_class in self.filters_path:
            suggested: Dict[str, Any] = {}

            for field_name, field_type, hint in iter_parameter_hints(filter_class):
                name = optuna_param_name(filter_class, field_name)

                if is_constant(hint):
                    suggested[field_name] = field_type(hint.lower_bound)
                elif field_type is int:
                    suggested[field_name] = trial.suggest_int(
                        name=name,
                        low=hint.lower_bound,
                        high=hint.upper_bound,
                        step=int(hint.step) if hint.step is not None else 1,
                    )
                elif field_type is float:
                    suggested[field_name] = trial.suggest_float(
                        name=name,
                        low=hint.lower_bound,
                        high=hint.upper_bound,
                        step=hint.step,
                    )

            parameters = filter_class.parameters_class(**suggested)  # type: ignore[attr-defined]
            optimized_filters.append(filter_class.parametrized(parameters))

        # Full pipeline: frozen filters first, then the ones under optimization
        pipeline = self.frozen_filters + optimized_filters

        # Store the full pipeline in the trial for logging
        trial.set_user_attr("filter_parameters", serialize_pipeline(pipeline))

        # Create a mini-dataset with the same samples to avoid modifying the original dataset
        mini_dataset = Dataset(samples=self.samples)

        # Use a temporary directory to stage the filtered dataset for evaluation
        with tempfile.TemporaryDirectory() as temp_dir_name:
            staging_path = Path(temp_dir_name) / "yolo_eval_dataset"

            staged_dataset = mini_dataset.apply_filters_and_export(
                filters=pipeline,
                destination_path=staging_path,
            )

            return self.model.evaluate(staged_dataset)


if __name__ == "__main__":
    from .filters import (
        ClaheFilterAdapter,
        LaplacianSharpenFilterAdapter,
        MedianBlurFilterAdapter,
        SaturationBoostFilterAdapter,
    )

    # Create or load an Optuna study for YOLO preprocessing optimization
    study = optuna.create_study(
        study_name="yolo_preprocessing_optimization_3",
        storage="sqlite:///yolo_preprocessing_optimization.db",
        load_if_exists=True,
        direction="maximize",
    )

    # Load dataset and pick random samples for training
    dataset = Dataset.from_yolo_directory(Path("resources/dataset"))
    samples = dataset.pick_random_samples(
        sample_count=100,
        domain="target",
        split=DatasetSplit.VAL,
    )
    model = Yolo.load_model(Path("resources/model.pt"))

    # Without frozen filters, all the filters in the path are optimized jointly
    trainer = Trainer(
        model=model,
        filters_path=[
            MedianBlurFilterAdapter,
            SaturationBoostFilterAdapter,
            ClaheFilterAdapter,
            LaplacianSharpenFilterAdapter,
        ],
        samples=samples,
    )

    # Run the optimization for 30 trials
    study.optimize(trainer.objective, n_trials=30)

    # Print the best hyperparameters found
    print(study.best_params)