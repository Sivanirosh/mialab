"""Configuration management for experiments.

This module handles experiment configurations, random forest parameters,
and ablation study setups.
"""

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class OptimizationLevel(Enum):
    """Random Forest optimization levels."""
    NONE = "none"
    QUICK = "quick" 
    FULL = "full"


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing parameters."""
    skullstrip_pre: bool = True
    normalization_pre: bool = True
    registration_pre: bool = True
    coordinates_feature: bool = True
    intensity_feature: bool = True
    gradient_intensity_feature: bool = True


@dataclass
class PostprocessingConfig:
    """Configuration for postprocessing parameters."""
    simple_post: bool = True


@dataclass
class RandomForestConfig:
    """Configuration for Random Forest parameters."""
    n_estimators: int = 100
    max_depth: Optional[int] = None
    max_features: str = 'sqrt'
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    bootstrap: bool = True
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    description: str
    preprocessing: PreprocessingConfig
    postprocessing: PostprocessingConfig
    forest: RandomForestConfig
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convert dictionaries back to dataclasses
        preprocessing = PreprocessingConfig(**config_dict['preprocessing'])
        postprocessing = PostprocessingConfig(**config_dict['postprocessing'])
        forest = RandomForestConfig(**config_dict['forest'])
        
        return cls(
            name=config_dict['name'],
            description=config_dict['description'],
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            forest=forest
        )
    

    def to_pipeline_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary format expected by the pipeline."""
        
        def safe_asdict(obj):
            if isinstance(obj, list):
                # take first item if wrapped in a list
                if len(obj) > 0 and is_dataclass(obj[0]):
                    return asdict(obj[0])
                else:
                    return obj
            elif is_dataclass(obj):
                return asdict(obj)
            return obj
        
        return {
            'preprocessing': safe_asdict(self.preprocessing),
            'postprocessing': safe_asdict(self.postprocessing),
            'forest': safe_asdict(self.forest)
        }


class RandomForestOptimizer:
    """Handles Random Forest hyperparameter optimization."""
    
    @staticmethod
    def get_default_parameters() -> RandomForestConfig:
        """Get default Random Forest parameters."""
        return RandomForestConfig()
    
    @staticmethod
    def get_parameter_grid(optimization_level: OptimizationLevel) -> Dict[str, List]:
        """Get parameter grid for hyperparameter optimization."""
        if optimization_level == OptimizationLevel.NONE:
            return {}
        
        elif optimization_level == OptimizationLevel.QUICK:
            return {
                'n_estimators': [50, 100],
                'max_depth': [10, 15, 20],
                'max_features': ['sqrt', 'log2'],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 5]
            }
        
        elif optimization_level == OptimizationLevel.FULL:
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, 25],
                'max_features': ['sqrt', 'log2', None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4, 5],
                'bootstrap': [True, False]
            }
        
        else:
            raise ValueError(f"Unknown optimization level: {optimization_level}")
    
    @staticmethod
    def estimate_optimization_time(optimization_level: OptimizationLevel, 
                                 n_samples: int = 10000) -> str:
        """Estimate optimization time based on level and data size."""
        if optimization_level == OptimizationLevel.NONE:
            return "< 1 minute"
        
        grid = RandomForestOptimizer.get_parameter_grid(optimization_level)
        n_combinations = 1
        for values in grid.values():
            n_combinations *= len(values)
        
        # Rough time estimates based on grid size and data size
        base_time_per_combination = 30  # seconds
        if n_samples > 50000:
            base_time_per_combination *= 2
        
        total_seconds = n_combinations * base_time_per_combination
        total_minutes = total_seconds / 60
        
        if total_minutes < 60:
            return f"~{int(total_minutes)} minutes"
        else:
            return f"~{total_minutes/60:.1f} hours"


class AblationStudyConfigurator:
    """Creates configurations for ablation study experiments."""
    
    @staticmethod
    def create_ablation_configs(forest_config: RandomForestConfig) -> Dict[int, ExperimentConfig]:
        """Create all 9 ablation study configurations."""
        
        configs = {}
        
        # Experiment 0: Baseline (no preprocessing)
        configs[0] = ExperimentConfig(
            name="baseline_none",
            description="Baseline without any preprocessing",
            preprocessing=PreprocessingConfig(
                skullstrip_pre=False,
                normalization_pre=False,
                registration_pre=False,
                coordinates_feature=True,
                intensity_feature=True,
                gradient_intensity_feature=True
            ),
            postprocessing=PostprocessingConfig(simple_post=False),
            forest=forest_config
        )
        
        # Experiment 1: Normalization only
        configs[1] = ExperimentConfig(
            name="normalization_only",
            description="With normalization only",
            preprocessing=PreprocessingConfig(
                skullstrip_pre=False,
                normalization_pre=True,
                registration_pre=False,
                coordinates_feature=True,
                intensity_feature=True,
                gradient_intensity_feature=True
            ),
            postprocessing=PostprocessingConfig(simple_post=False),
            forest=forest_config
        )
        
        # Experiment 2: Skull stripping only
        configs[2] = ExperimentConfig(
            name="skullstrip_only",
            description="With skull stripping only",
            preprocessing=PreprocessingConfig(
                skullstrip_pre=True,
                normalization_pre=False,
                registration_pre=False,
                coordinates_feature=True,
                intensity_feature=True,
                gradient_intensity_feature=True
            ),
            postprocessing=PostprocessingConfig(simple_post=False),
            forest=forest_config
        )
        
        # Experiment 3: Registration only
        configs[3] = ExperimentConfig(
            name="registration_only",
            description="With registration only",
            preprocessing=PreprocessingConfig(
                skullstrip_pre=False,
                normalization_pre=False,
                registration_pre=True,
                coordinates_feature=True,
                intensity_feature=True,
                gradient_intensity_feature=True
            ),
            postprocessing=PostprocessingConfig(simple_post=False),
            forest=forest_config
        )
        
        # Experiment 4: Normalization + Skull stripping
        configs[4] = ExperimentConfig(
            name="norm_skull",
            description="With normalization + skull stripping",
            preprocessing=PreprocessingConfig(
                skullstrip_pre=True,
                normalization_pre=True,
                registration_pre=False,
                coordinates_feature=True,
                intensity_feature=True,
                gradient_intensity_feature=True
            ),
            postprocessing=PostprocessingConfig(simple_post=False),
            forest=forest_config
        )
        
        # Experiment 5: Normalization + Registration
        configs[5] = ExperimentConfig(
            name="norm_reg",
            description="With normalization + registration",
            preprocessing=PreprocessingConfig(
                skullstrip_pre=False,
                normalization_pre=True,
                registration_pre=True,
                coordinates_feature=True,
                intensity_feature=True,
                gradient_intensity_feature=True
            ),
            postprocessing=PostprocessingConfig(simple_post=False),
            forest=forest_config
        )
        
        # Experiment 6: Registration + Skull stripping
        configs[6] = ExperimentConfig(
            name="reg_skull",
            description="With registration + skull stripping",
            preprocessing=PreprocessingConfig(
                skullstrip_pre=True,
                normalization_pre=False,
                registration_pre=True,
                coordinates_feature=True,
                intensity_feature=True,
                gradient_intensity_feature=True
            ),
            postprocessing=PostprocessingConfig(simple_post=False),
            forest=forest_config
        )
        
        # Experiment 7: All preprocessing
        configs[7] = ExperimentConfig(
            name="all_preprocessing",
            description="With normalization + skull stripping + registration",
            preprocessing=PreprocessingConfig(
                skullstrip_pre=True,
                normalization_pre=True,
                registration_pre=True,
                coordinates_feature=True,
                intensity_feature=True,
                gradient_intensity_feature=True
            ),
            postprocessing=PostprocessingConfig(simple_post=False),
            forest=forest_config
        )
        
        # Experiment 8: All preprocessing + postprocessing
        configs[8] = ExperimentConfig(
            name="all_preprocessing_postprocessing",
            description="With all preprocessing + postprocessing",
            preprocessing=PreprocessingConfig(
                skullstrip_pre=True,
                normalization_pre=True,
                registration_pre=True,
                coordinates_feature=True,
                intensity_feature=True,
                gradient_intensity_feature=True
            ),
            postprocessing=PostprocessingConfig(simple_post=True),
            forest=forest_config
        )
        
        return configs
    
    @staticmethod
    def get_experiment_summary() -> Dict[int, str]:
        """Get summary description of each experiment."""
        return {
            0: "Baseline (no preprocessing)",
            1: "Normalization only",
            2: "Skull stripping only", 
            3: "Registration only",
            4: "Normalization + Skull stripping",
            5: "Normalization + Registration",
            6: "Registration + Skull stripping", 
            7: "All preprocessing (Norm + Skull + Reg)",
            8: "All preprocessing + Post-processing"
        }


class ConfigurationValidator:
    """Validates experiment configurations."""
    
    @staticmethod
    def validate_paths(data_atlas_dir: str, data_train_dir: str, data_test_dir: str) -> List[str]:
        """Validate that required data paths exist."""
        missing_paths = []
        
        if not os.path.exists(data_atlas_dir):
            missing_paths.append(f"Atlas directory: {data_atlas_dir}")
        
        if not os.path.exists(data_train_dir):
            missing_paths.append(f"Training directory: {data_train_dir}")
        
        if not os.path.exists(data_test_dir):
            missing_paths.append(f"Test directory: {data_test_dir}")
        
        return missing_paths
    
    @staticmethod
    def validate_experiment_config(config: ExperimentConfig) -> List[str]:
        """Validate experiment configuration."""
        issues = []
        
        # Check name
        if not config.name or not config.name.strip():
            issues.append("Experiment name cannot be empty")
        
        # Check Random Forest parameters
        if config.forest.n_estimators <= 0:
            issues.append("n_estimators must be positive")
        
        if config.forest.max_depth is not None and config.forest.max_depth <= 0:
            issues.append("max_depth must be positive or None")
        
        if config.forest.min_samples_split < 2:
            issues.append("min_samples_split must be at least 2")
        
        if config.forest.min_samples_leaf < 1:
            issues.append("min_samples_leaf must be at least 1")
        
        # Check that at least one feature is enabled
        if not any([config.preprocessing.coordinates_feature,
                   config.preprocessing.intensity_feature,
                   config.preprocessing.gradient_intensity_feature]):
            issues.append("At least one feature type must be enabled")
        
        return issues
    
    @staticmethod
    def estimate_runtime(optimization_level: OptimizationLevel, 
                        n_experiments: int = 9,
                        n_subjects: int = 10) -> str:
        """Estimate total runtime for ablation study."""
        # Estimate optimization time
        opt_time_str = RandomForestOptimizer.estimate_optimization_time(optimization_level)
        
        # Estimate per-experiment time (rough estimate)
        per_exp_minutes = 5 + (n_subjects * 2)  # Base time + time per subject
        total_exp_minutes = n_experiments * per_exp_minutes
        
        # Parse optimization time
        if "minute" in opt_time_str:
            opt_minutes = int(opt_time_str.split()[0].replace("~", ""))
        elif "hour" in opt_time_str:
            opt_hours = float(opt_time_str.split()[0].replace("~", ""))
            opt_minutes = opt_hours * 60
        else:
            opt_minutes = 0
        
        total_minutes = opt_minutes + total_exp_minutes
        
        if total_minutes < 60:
            return f"~{int(total_minutes)} minutes"
        else:
            return f"~{total_minutes/60:.1f} hours"


def create_default_config(name: str = "default", 
                         optimization_level: OptimizationLevel = OptimizationLevel.QUICK) -> ExperimentConfig:
    """Create a default experiment configuration."""
    
    if optimization_level == OptimizationLevel.NONE:
        forest_config = RandomForestOptimizer.get_default_parameters()
    else:
        # Use reasonable defaults for optimized versions
        forest_config = RandomForestConfig(
            n_estimators=100,
            max_depth=15,
            max_features='sqrt',
            min_samples_split=5,
            min_samples_leaf=2
        )
    
    return ExperimentConfig(
        name=name,
        description=f"Default configuration with {optimization_level.value} optimization",
        preprocessing=PreprocessingConfig(),
        postprocessing=PostprocessingConfig(),
        forest=forest_config
    )