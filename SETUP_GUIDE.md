# OpenKBP Setup Guide

## Overview

OpenKBP is an AAPM Grand Challenge for dose prediction in radiotherapy. This guide documents the setup process for running the dose prediction model locally.

## Project Structure

```
OpenKBP Project/
├── open-kbp/                    # Dose prediction (Stage 1)
│   ├── provided-data/
│   │   ├── train-pats/          # 200 patients
│   │   ├── validation-pats/     # 40 patients
│   │   └── test-pats/           # 100 patients
│   ├── provided_code/
│   ├── results/
│   │   └── baseline/
│   │       ├── models/
│   │       ├── validation-predictions/
│   │       └── submissions/
│   ├── main.py
│   ├── local_notebook.ipynb     # VSCode-friendly notebook
│   └── venv/                    # Virtual environment
```

## Bugs Fixed

### 1. Outdated Keras Imports (network_architectures.py)

**Before:**
```python
from keras.engine.keras_tensor import KerasTensor
from keras.layers import Activation, AveragePooling3D, Conv3D, ...
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.models import Model
from keras.optimizers.optimizer_v2.optimizer_v2 import OptimizerV2
```

**After:**
```python
from typing import Optional, Any
from tensorflow.keras.layers import (
    Activation, AveragePooling3D, Conv3D, Conv3DTranspose,
    Input, LeakyReLU, SpatialDropout3D, Concatenate, BatchNormalization
)
from tensorflow.keras.models import Model

KerasTensor = Any
OptimizerV2 = Any
```

### 2. Outdated Keras Imports (network_functions.py)

**Before:**
```python
from keras.models import load_model
from keras.optimizers.optimizer_v2.adam import Adam
```

**After:**
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
```

### 3. Deprecated `concatenate` Function

**Before:**
```python
x = concatenate([ct_image, roi_masks])
```

**After:**
```python
x = Concatenate()([ct_image, roi_masks])
```

### 4. Optimizer State Incompatibility (Keras 3)

Added recompile after loading saved models in `network_functions.py`:

```python
def initialize_networks(self) -> None:
    if self.current_epoch >= 1:
        self.generator = load_model(self._get_generator_path(self.current_epoch))
        # Recompile to fix optimizer state (Keras 3 compatibility)
        self.generator.compile(loss="mean_absolute_error", optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999))
    else:
        self.generator = self.define_generator()
```

### 5. Missing Dependency

```bash
pip install more_itertools
```

## Setup Steps

### 1. Clone Repository

```bash
cd "/Users/neiltripathi/Documents/OpenKBP Project"
git clone https://github.com/ababier/open-kbp.git
```

### 2. Create Virtual Environment

```bash
cd open-kbp
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
pip install tensorflow numpy pandas scipy matplotlib seaborn tqdm h5py scikit-learn more_itertools
```

### 4. Register Jupyter Kernel (for VSCode/Jupyter)

```bash
pip install ipykernel
python -m ipykernel install --user --name=openkbp --display-name="OpenKBP"
```

## Running the Code

### Terminal

```bash
cd "/Users/neiltripathi/Documents/OpenKBP Project/open-kbp"
source venv/bin/activate
python main.py
```

### Jupyter (Browser)

```bash
cd "/Users/neiltripathi/Documents/OpenKBP Project/open-kbp"
source venv/bin/activate
jupyter lab
```

### VSCode

1. Open `local_notebook.ipynb`
2. Select Kernel → Python Environments → `venv/bin/python`
3. Run cells

## Configuration

Edit `main.py` to change:

```python
prediction_name = "baseline"  # Model name
num_epochs = 2                # Increase to 100-200 for real training
test_time = False             # Set True to evaluate on test set
```

To improve model quality, edit `provided_code/network_functions.py` line 25:

```python
initial_number_of_filters=64,  # Was 1, increase to 64+
```

## Results (Baseline, 2 Epochs)

| Metric | Score |
|--------|-------|
| DVH Score | 50.026 |
| Dose Score | 25.337 |

Lower is better. Winning models achieved DVH ~1.5, Dose ~2.4.

## Training Time Estimates

| Epochs | Time (CPU) |
|--------|------------|
| 2 | ~10 min |
| 100 | ~7 hours |
| 200 | ~14 hours |

## Output Files

```
results/baseline/
├── models/epoch_*.h5           # Saved model weights
├── train-predictions/          # Training set predictions
├── validation-predictions/     # Validation set predictions
└── submissions/baseline.zip    # Ready for CodaLab submission
```

## Useful Links

- [OpenKBP GitHub](https://github.com/ababier/open-kbp)
- [OpenKBP-Opt (Optimizer)](https://github.com/ababier/open-kbp-opt)
- [OpenKBP Paper](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.14845)
- [AAPM Grand Challenge](https://www.aapm.org/GrandChallenge/OpenKBP/)