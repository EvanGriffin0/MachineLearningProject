# Workspace

Project workspace for an image classification pipeline built with TensorFlow/Keras.

## Directory Structure

```
Workspace/
├── notebooks/          # Jupyter notebooks for experimentation, EDA, and reporting
├── src/
│   ├── data/           # Dataset loading and preprocessing utilities
│   ├── models/         # Model architecture definitions (CNN, transfer learning, etc.)
│   ├── training/       # Training loop and configuration helpers
│   └── evaluation/     # Evaluation metrics and reporting utilities
└── outputs/
    ├── figures/        # Saved plots and visualisations generated during experiments
    └── models/         # Saved model weights and checkpoints
```

## Folder Descriptions

| Folder | Purpose |
|---|---|
| `notebooks/` | Interactive notebooks used for exploration, prototyping, and presenting results. |
| `src/data/` | Functions to load and preprocess image datasets from disk. |
| `src/models/` | Keras model builders — CNN from scratch and transfer learning wrappers. |
| `src/training/` | `train_model()` helper and any training-related utilities. |
| `src/evaluation/` | Accuracy, precision, recall, F1, and confusion matrix helpers. |
| `outputs/figures/` | Matplotlib/Seaborn figures saved by notebooks or evaluation scripts. |
| `outputs/models/` | Serialised Keras models (`.keras` / `.h5`) and checkpoint files. |

## Setup

```bash
pip install -r requirements.txt
```
