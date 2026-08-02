"""
trainer.py
This module provides a Trainer class for optimizing sequences of image preprocessing filters
using Optuna for hyperparameter search. It leverages Pydantic for parameter validation and
type safety, and supports flexible filter pipelines. The Trainer applies a series of filters
to image samples, simulates evaluation (e.g., YOLO accuracy), and integrates with Optuna to
find optimal hyperparameters.

Main Components:
- Trainer: Manages filter pipelines, sample processing, and optimization tasks.
- Task: Represents a single optimization iteration and its parameters.

Each class and method is documented, and all Pydantic fields include descriptions for clarity.
"""

from pathlib import Path
from typing import Dict, List, Type

import optuna
from optuna import Trial
from pydantic import BaseModel
from tqdm import tqdm

from .dataset import (
    Dataset,
    DatasetSplit,
    StoredSample,
    TransientSample,
)
from .filters import (
    FilterAdapter,
    FilterParametersHint,
    FilterType,
    NoOpFilterAdapter,
)
from .yolo import Yolo


class Trainer:
    """
    Trainer class for optimizing a sequence of image preprocessing filters
    using Optuna for hyperparameter search.
    """

    model: Yolo
    filters_path: List[Type[FilterAdapter]]
    samples: List[StoredSample]

    def __init__(
        self,
        model: Yolo,
        filters_path: List[Type[FilterAdapter]],
        samples: List[StoredSample],
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            filters_path: List of filter adapter classes in the order to be applied.
            dataset: List of samples to process.
        """
        # Ensure filters are ordered correctly while allowing skipped layers (NoOp)
        last_filter_type_value = 0
        for i, filter_class in enumerate(filters_path):
            if filter_class.filter_type == FilterType.NO_OP:
                continue
            if filter_class.filter_type <= last_filter_type_value:
                raise ValueError(
                    f"Violation of ordering constraint at layer {i}: found '{filter_class.filter_type.name}'."
                )
            last_filter_type_value = filter_class.filter_type

        # Ensure not all filters are NO_OP
        if all([filter == NoOpFilterAdapter for filter in filters_path]):
            raise ValueError("The filter path cannot consist solely of NO_OP filters.")

        self.model = model
        self.filters_path = filters_path
        self.samples = samples

    def objective(self, trial: Trial) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object.

        Returns:
            YOLO accuracy (float).
        """
        filter_parameters: List[BaseModel] = []
        filter_parameters_serializable: Dict[str, Dict] = {}

        # Suggest hyperparameters for each filter in the path
        for idx, filter_class in enumerate(self.filters_path):
            filter_parameters_class = filter_class.parameters_class  # type: ignore[attr-defined]
            filter_suggested_parameters = {}

            # Iterate over each hyperparameter field in the filter's parameter class
            for field_name, field_info in filter_parameters_class.model_fields.items():
                field_type = field_info.annotation
                type_hints = filter_parameters_class.__annotations__[field_name]

                # Extract hyperparameter hints from type metadata
                for field_hint in type_hints.__metadata__:
                    if isinstance(field_hint, FilterParametersHint):
                        # Suggest integer hyperparameters
                        if field_type is int:
                            filter_suggested_parameters[field_name] = trial.suggest_int(
                                name=f"{filter_class.name}_{field_name}",
                                low=field_hint.lower_bound,
                                high=field_hint.upper_bound,
                                step=(
                                    field_hint.step
                                    if field_hint.step is not None
                                    else 1
                                ),
                            )
                        # Suggest float hyperparameters
                        if field_type is float:
                            filter_suggested_parameters[field_name] = (
                                trial.suggest_float(
                                    name=f"{filter_class.name}_{field_name}",
                                    low=field_hint.lower_bound,
                                    high=field_hint.upper_bound,
                                    step=field_hint.step,
                                )
                            )

            # Store the suggested parameters for this filter (preserve order)
            params = filter_class.parameters_class(  # type: ignore[attr-defined]
                **filter_suggested_parameters
            )
            filter_parameters.append(params)
            filter_parameters_serializable[f"{filter_class.name}_{idx}"] = (
                params.model_dump()
            )

        # Store filter parameters in trial user attributes for logging
        trial.set_user_attr("filter_parameters", filter_parameters_serializable)

        # Apply the filter pipeline to each image in the dataset
        transformed_samples: List[TransientSample] = []
        for sample in tqdm(self.samples, desc="Applying filters", unit="img"):
            image = sample.load_image()
            for layer_index, filter_class in enumerate(self.filters_path):
                parametrized_filter = filter_class.parametrized(
                    filter_parameters[layer_index]
                )
                image = parametrized_filter.apply(image)
            transformed_samples.append(
                TransientSample(numpy_image=image, labels=sample.labels)
            )

        transient_dataset = Dataset.create_transient_dataset(transformed_samples)
        staging_dataset = transient_dataset.to_staging_dataset()

        return self.model.evaluate(staging_dataset)


if __name__ == "__main__":
    from filters import (
        ClaheFilterAdapter,
        FilterParametersHint,
        LaplacianSharpenFilterAdapter,
        MedianBlurFilterAdapter,
        NoOpFilterAdapter,
        SaturationBoostFilterAdapter,
    )

    # Create or load an Optuna study for YOLO preprocessing optimization
    study = optuna.create_study(
        study_name="yolo_preprocessing_optimization_3",
        storage="sqlite:///yolo_preprocessing_optimization.db",
        load_if_exists=True,
        direction="maximize",
    )

    dataset = Dataset.load_from_directory(Path("resources/dataset"))
    samples = dataset.pick_random_samples(
        domain="target", split=DatasetSplit.VAL
    )
    model = Yolo.load_model(Path("resources/model.pt"))

    # Instantiate the Trainer with a sequence of filter adapters and a random dataset
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

    # Run the optimization for 100 trials
    study.optimize(trainer.objective, n_trials=30)

    # Print the best hyperparameters found
    print(study.best_params)
