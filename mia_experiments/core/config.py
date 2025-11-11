"""Configuration utilities for the single-run experiment workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class OptimizationLevel(Enum):
    """Random Forest optimisation presets."""

    NONE = "none"
    QUICK = "quick"
    FULL = "full"


@dataclass
class PreprocessingConfig:
    """Configuration flags for preprocessing components."""

    skullstrip_pre: bool = True
    normalization_pre: bool = True
    registration_pre: bool = True
    biascorrection_pre: bool = False
    coordinates_feature: bool = True
    intensity_feature: bool = True
    gradient_intensity_feature: bool = True


@dataclass
class PostprocessingConfig:
    """Configuration parameters for postprocessing."""

    simple_post: bool = True
    copy_unhandled_labels: bool = True
    recipes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Legacy fields kept for backwards compatibility with older config files.
    min_component_size: int = 5
    per_tissue_sizes: Optional[Dict[str, int]] = None
    kernel_radius: Tuple[int, int, int] = (1, 1, 1)
    retention_threshold: float = 0.5


@dataclass
class RandomForestConfig:
    """Random Forest hyper-parameter configuration."""

    n_estimators: int = 100
    max_depth: Optional[int] = None
    max_features: Optional[str] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    bootstrap: bool = True
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class ExperimentConfig:
    """Full configuration payload passed to the pipeline."""

    name: str
    description: str
    preprocessing: PreprocessingConfig
    postprocessing: PostprocessingConfig
    forest: RandomForestConfig

    def save(self, filepath: str) -> None:
        """Persist the configuration to disk."""

        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ExperimentConfig":
        """Reconstruct an experiment configuration from disk."""

        with open(filepath, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        return cls(
            name=payload["name"],
            description=payload["description"],
            preprocessing=PreprocessingConfig(**payload["preprocessing"]),
            postprocessing=PostprocessingConfig(**payload["postprocessing"]),
            forest=RandomForestConfig(**payload["forest"]),
        )

    def to_pipeline_dict(self) -> Dict[str, Any]:
        """Serialise the configuration for `pipeline.main`."""

        def _flatten(value: Any) -> Any:
            if is_dataclass(value):
                return asdict(value)
            if isinstance(value, list) and value and is_dataclass(value[0]):
                return asdict(value[0])
            return value

        return {
            "preprocessing": _flatten(self.preprocessing),
            "postprocessing": _flatten(self.postprocessing),
            "forest": _flatten(self.forest),
        }


class RandomForestOptimizer:
    """Helpers for deriving Random Forest configurations."""

    @staticmethod
    def get_default_parameters() -> RandomForestConfig:
        """Return a sensible default configuration."""

        return RandomForestConfig()

    @staticmethod
    def get_parameter_grid(optimization_level: OptimizationLevel) -> Dict[str, list]:
        """Return the hyper-parameter grid for a requested optimisation level."""

        if optimization_level == OptimizationLevel.NONE:
            return {}

        if optimization_level == OptimizationLevel.QUICK:
            return {
                "n_estimators": [50, 100],
                "max_depth": [10, 15, 20],
                "max_features": [None],
                "min_samples_split": [5, 10],
                "min_samples_leaf": [2, 5],
            }

        if optimization_level == OptimizationLevel.FULL:
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 15, 20, 25],
                "max_features": [None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4, 5],
                "bootstrap": [True, False],
            }

        raise ValueError(f"Unknown optimisation level: {optimization_level}")

    @staticmethod
    def estimate_optimization_time(
        optimization_level: OptimizationLevel,
        n_samples: int = 10_000,
    ) -> str:
        """Rudimentary wall-clock estimate for grid-search style optimisation."""

        if optimization_level == OptimizationLevel.NONE:
            return "< 1 minute"

        grid = RandomForestOptimizer.get_parameter_grid(optimization_level)
        n_combinations = 1
        for values in grid.values():
            n_combinations *= len(values)

        base_seconds = 30 if n_samples <= 50_000 else 60
        total_minutes = (n_combinations * base_seconds) / 60

        if total_minutes < 60:
            return f"~{int(total_minutes)} minutes"

        return f"~{total_minutes/60:.1f} hours"


def create_all_preprocessing_config(
    name: str = "all_preprocessing",
    include_postprocessing: bool = True,
    forest_config: Optional[RandomForestConfig] = None,
    description: Optional[str] = None,
) -> ExperimentConfig:
    """Factory for the canonical experiment configuration.

    The resulting configuration enables every preprocessing component and toggles
    post-processing based on `include_postprocessing`. The pipeline will still
    emit both pre- and post-processed metrics, allowing downstream analysis to
    separate them without re-running the experiment.
    """

    if forest_config is None:
        forest_config = RandomForestOptimizer.get_default_parameters()

    preprocessing = PreprocessingConfig(
        skullstrip_pre=True,
        normalization_pre=True,
        registration_pre=True,
        biascorrection_pre=False,
        coordinates_feature=True,
        intensity_feature=True,
        gradient_intensity_feature=True,
    )

    postprocessing = PostprocessingConfig(simple_post=include_postprocessing)

    experiment_description = description or (
        "All preprocessing components enabled{}.".format(
            " with postprocessing" if include_postprocessing else " without postprocessing"
        )
    )

    return ExperimentConfig(
        name=name,
        description=experiment_description,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        forest=forest_config,
    )


def create_default_config(
    name: str = "all_preprocessing",
    include_postprocessing: bool = True,
    forest_config: Optional[RandomForestConfig] = None,
) -> ExperimentConfig:
    """Backwards compatible helper that maps to the new single-run setup."""

    return create_all_preprocessing_config(
        name=name,
        include_postprocessing=include_postprocessing,
        forest_config=forest_config,
    )
