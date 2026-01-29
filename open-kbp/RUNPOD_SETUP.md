# RunPod Training Setup

## Quick Start

### 1. Create Pod
- Template: **PyTorch** (includes TensorFlow)
- GPU: RTX 3090 or RTX 4090 (24GB VRAM)
- Disk: 50GB (for data + models)

### 2. Clone Repo
```bash
cd /workspace
git clone https://github.com/neilt93/OpenKBP-Project.git
cd OpenKBP-Project/open-kbp
```

### 3. Install Dependencies
```bash
pip install tensorflow numpy pandas scipy matplotlib tqdm h5py scikit-learn more_itertools
```

### 4. Upload Data
The `provided-data/` folder (~2GB) is not in git. Transfer it using one of:

**Option A: runpodctl (recommended for large files)**
```bash
# On your local machine:
runpodctl send provided-data.zip

# On RunPod (copy the command it gives you):
runpodctl receive <code>
unzip provided-data.zip -d /workspace/OpenKBP-Project/open-kbp/
```

**Option B: Direct from original repo**
```bash
cd /workspace
git clone https://github.com/ababier/open-kbp.git open-kbp-data
cp -r open-kbp-data/provided-data /workspace/OpenKBP-Project/open-kbp/
rm -rf open-kbp-data
```

### 5. Run Training
```bash
cd /workspace/OpenKBP-Project/open-kbp

# Default: 64 filters, 100 epochs
python runpod_train.py

# Custom configuration
python runpod_train.py --filters 64 --epochs 100 --save-freq 10
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--filters` | 64 | Initial U-Net filters |
| `--epochs` | 100 | Training epochs |
| `--save-freq` | 10 | Save every N epochs |
| `--keep-history` | 5 | Keep last N models |
| `--predict-only` | false | Skip training, run predictions |

## Expected Results

| Config | DVH Score | Dose Score | Time (3090) |
|--------|-----------|------------|-------------|
| 16f/5e | 21.4 | 11.7 | ~5 min |
| 64f/100e | ~5-10 | ~3-5 | ~1-2 hr |
| Winning | ~1.5 | ~2.4 | - |

## Download Results

After training, get your models:
```bash
# On RunPod:
cd /workspace/results
runpodctl send 64filter_100epoch/

# On local:
runpodctl receive <code>
```

## Troubleshooting

**OOM Error**: Reduce batch size in `provided_code/data_loader.py` or use smaller filters.

**No GPU**: Check with `nvidia-smi`. Ensure TensorFlow sees GPU:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```
