# Medical Image Analysis Laboratory

Welcome to the medical image analysis laboratory (MIALab).
This repository contains all code you will need to get started with classical medical image analysis.

During the MIALab you will work on the task of brain tissue segmentation from magnetic resonance (MR) images.
We have set up an entire pipeline to solve this task, specifically:

- Pre-processing
- Registration
- Feature extraction
- Voxel-wise tissue classification
- Post-processing
- Evaluation

## Tissue-aware post-processing

The default pipeline now applies label-specific morphological cleanup and connected-component filtering after the
voxel-wise classifier. Each tissue class (white matter, grey matter, hippocampus, amygdala, thalamus) can be assigned
custom heuristics directly via the experiment configuration (`postprocessing.recipes`). Per-label recipes control
opening/closing radii, the number of connected components to retain, and minimum voxel counts, allowing the post-processed
segmentation to remain anatomically plausible while suppressing noise. When no overrides are provided, sensible defaults
are applied automatically.

## Automated experiment reporting

Once a run finishes (e.g., under `experiments_001/<experiment_id>/<timestamp>/`), you can generate plots and 3D
screenshots summarising the metrics and segmentations:

```
python generate_experiment_report.py \
    --experiment-dir experiments_001/all_preprocessing_YYYYMMDD_HHMMSS/<timestamped_run> \
    --max-screenshots 3
```

The script writes all figures and tables to `<experiment_dir>/report/` by default. Screenshots are rendered off-screen
using VTK, so a GUI session is not required. Matplotlib and seaborn are used for plotting; install them via pip if they
are not already available in your environment.

After you complete the exercises, dive into the 
    
    pipeline.py 

script to learn how all of these steps work together. 

During the laboratory you will get to know the entire pipeline and investigate one of these pipeline elements in-depth.
You will get to know and to use various libraries and software tools needed in the daily life as biomedical engineer or researcher in the medical image analysis domain.

Enjoy!

----

Found a bug or do you have suggestions? Open an issue or better submit a pull request.
