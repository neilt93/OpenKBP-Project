# RunPod Training Setup

## Quick Start

### 1. Create Pod
- Template: **PyTorch** (includes TensorFlow)
- GPU: RTX 3090 or RTX 4090 (24GB VRAM)
- Disk: 50GB (for data + models)

### 2. Clone Original Repo (has data)
```bash
cd /workspace
git clone https://github.com/ababier/open-kbp.git
cd open-kbp
```

### 3. Upload Code Update from Local Machine

**On your Mac:**
```bash
cd "/Users/neiltripathi/Documents/OpenKBP Project/open-kbp-modified"
runpodctl send code-update.zip
```

**On RunPod (use the code it gives you):**
```bash
cd /workspace/open-kbp
runpodctl receive <CODE>
python -c "import zipfile; zipfile.ZipFile('code-update.zip').extractall('.')"
```

### 4. Install Dependencies
```bash
# IMPORTANT: Use TensorFlow 2.16.1 to match RunPod's cuDNN 9.1.0
pip uninstall tensorflow -y 2>/dev/null
pip install pandas numpy scipy tensorflow==2.16.1 tqdm more_itertools
```

### 5. Run Training
```bash
cd /workspace/open-kbp

# Full optimized config (use --no-jit to avoid cuDNN version issues)
python runpod_train.py --filters 64 --epochs 100 --use-se --use-dvh --use-aug --batch-size 2 --no-jit

# Baseline (matches original Colab)
python test_baseline.py --filters 64 --epochs 100
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

**cuDNN Version Mismatch** (e.g., `Loaded runtime CuDNN library: 9.1.0 but source was compiled with: 9.3.0`):
```bash
pip uninstall tensorflow -y
pip install tensorflow==2.16.1
```

**OOM Error**: Reduce batch size or use smaller filters:
```bash
python runpod_train.py --filters 32 --batch-size 1 --no-jit
```

**No GPU**: Check with `nvidia-smi`. Ensure TensorFlow sees GPU:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```
