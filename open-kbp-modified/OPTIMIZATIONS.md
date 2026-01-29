# OpenKBP Training Optimizations

Summary of changes made to speed up training on RunPod GPU (RTX 3090).

---

## Benchmark Results

### Speed Progression
| Stage | Batch Time | Epoch Time | 100 Epochs | Notes |
|-------|------------|------------|------------|-------|
| Initial (CPU) | 24s | ~40 min | ~67 hours | TensorFlow not using GPU |
| GPU enabled | ~1.4s | ~2.3 min | ~4 hours | After `tensorflow[and-cuda]` |
| + Caching | ~1.0s | ~1.7 min | ~2.8 hours | Pre-shaped tensor cache |
| + XLA + Mixed Precision | ~0.65s | ~65s | ~1.8 hours | First epoch slower (XLA compile) |
| + Batch size 4 (slower!) | ~1.44s | ~72s | ~2.0 hours | Memory bandwidth limited |

**Optimal config: batch_size=2 with XLA + Mixed Precision**

### Batch Size Comparison (Measured)
| Batch Size | Batches/Epoch | Time/Batch | Epoch Time | Notes |
|------------|---------------|------------|------------|-------|
| 2 | 100 | 0.65s | **65s** | Optimal for RTX 3090 |
| 4 | 50 | 1.44s | 72s | 10% slower due to memory bandwidth |

**Why larger batch is slower:** 3D convolutions are memory-bandwidth limited, not compute limited. Doubling batch size requires >2x memory transfers, causing net slowdown.

### Training Loss Progression (64 filters, batch_size=4)
| Epoch | Loss | Notes |
|-------|------|-------|
| 0 | 1.44 | XLA compilation warmup |
| 1 | 1.23 | |
| 9 | 0.74 | |
| 100 | ~0.55 | Estimated final |

### GPU Metrics (RTX 3090)
| Metric | Value | Notes |
|--------|-------|-------|
| VRAM Usage (batch=2) | 5GB / 24GB | 21% |
| VRAM Usage (batch=4) | 9GB / 24GB | 38% |
| GPU Utilization | 35% | Memory-bound, not compute-bound |
| Memory Bandwidth | 936 GB/s | The bottleneck |

---

## Why GPU Utilization is Low

3D medical imaging with 128³ volumes is **memory-bandwidth limited**:

1. Each Conv3D layer reads ~2GB of activations from VRAM
2. GPU cores compute faster than memory can feed them
3. 35% utilization means 65% of the time cores wait for data
4. Larger batches don't help - they just move more data through the same bottleneck

**Solutions:**
- Use A100/H100 (2-3x higher memory bandwidth)
- Reduce input resolution (64³ instead of 128³)
- Accept the limitation for 3D workloads

---

## Optimizations Applied

### 1. GPU Detection Fix
**File:** `runpod_train.py`
```python
# Allow memory growth to avoid OOM
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```
**RunPod install:** `pip install tensorflow[and-cuda]`

### 2. Mixed Precision (FP16)
**File:** `runpod_train.py`
```python
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```
**File:** `provided_code/network_architectures.py`
```python
# Final layers stay float32 for numerical stability
Conv3DTranspose(..., dtype="float32")
AveragePooling3D(..., dtype="float32")
Activation("relu", dtype="float32")
```
**Benefit:** ~2x speedup on Tensor Cores

### 3. XLA JIT Compilation
**File:** `provided_code/network_architectures.py`
```python
generator.compile(loss="mean_absolute_error", optimizer=self.gen_optimizer, jit_compile=True)
```
**First epoch:** ~3 min XLA compile overhead
**Subsequent epochs:** Faster due to fused kernels

### 4. Pre-Stacked Data Arrays
**File:** `provided_code/data_loader.py`

**Before (slow):**
```python
for patient in batch:
    for key in required_files:
        batch_data.set_values(key, idx, cache[patient][key])
# 8 Python loops per batch
```

**After (fast):**
```python
# Init: stack all 200 patients into contiguous array
self._stacked_data[key] = np.stack([all patients], axis=0)
# Shape: (200, 128, 128, 128, channels)

# Training: single NumPy slice
batch_data = self._stacked_data[key][indices]
# Executed in C, not Python
```
**Benefit:** ~10x faster batch preparation
**Cost:** ~40GB RAM for 200 patients

### 5. Reduced Logging
**File:** `provided_code/network_functions.py`
```python
# Before: print every batch (100x per epoch)
# After: print once per epoch
print(f"Epoch {self.current_epoch}: avg_loss={avg_loss:.4f}")
```

### 6. Keras Format (no HDF5 warning)
**File:** `provided_code/network_functions.py`
```python
# Changed from .h5 to .keras
return self.model_dir / f"epoch_{epoch}.keras"
```

---

## Configuration

### Optimal Settings (RTX 3090)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Filters | 64 | 117M params |
| Epochs | 100 | |
| **Batch Size** | **2** | Optimal for memory-bound workload |
| Learning Rate | 0.0002 | |
| Optimizer | Adam (beta1=0.5, beta2=0.999) | |
| Loss | Mean Absolute Error | |
| Save Frequency | Every 10 epochs | |
| Keep History | Last 5 models | |

### Model Architecture
- **Type:** 3D U-Net
- **Input:** CT (128×128×128×1) + Structure Masks (128×128×128×10)
- **Output:** Dose (128×128×128×1)
- **Parameters:** 117M (446 MB)

---

## GPU Scaling Guide

| GPU | VRAM | Bandwidth | Expected Epoch Time | Best For |
|-----|------|-----------|---------------------|----------|
| RTX 3090 | 24GB | 936 GB/s | ~65s (64 filters) | 64 filters, batch=2 |
| RTX 4090 | 24GB | 1008 GB/s | ~60s | 64 filters, batch=2 |
| A100 40GB | 40GB | 1555 GB/s | ~40s | 128 filters, batch=2 |
| A100 80GB | 80GB | 2039 GB/s | ~30s | 128+ filters |
| H100 | 80GB | 3350 GB/s | ~20s | 256 filters |

**For larger models (128+ filters):** Use A100/H100, keep batch_size=1-2

---

## RunPod Commands

```bash
# Install TensorFlow with CUDA
pip install tensorflow[and-cuda]

# Run training (optimal settings)
python runpod_train.py --filters 64 --epochs 100

# Resume from checkpoint
python runpod_train.py --filters 64 --epochs 100

# Predict only (skip training)
python runpod_train.py --filters 64 --epochs 100 --predict-only

# Set optimal batch size
sed -i 's/batch_size: int = [0-9]*/batch_size: int = 2/' provided_code/data_loader.py

# Transfer files to RunPod
runpodctl send provided_code/data_loader.py
runpodctl send provided_code/network_functions.py
runpodctl send provided_code/network_architectures.py
runpodctl send runpod_train.py

# Receive files from RunPod
runpodctl receive <CODE>

# Check training progress (if disconnected)
ls -la /workspace/results/64filter_100epoch/models/
ps aux | grep python
nvidia-smi
```

---

## Files Modified

| File | Changes |
|------|---------|
| `runpod_train.py` | GPU config, mixed precision |
| `provided_code/data_loader.py` | Pre-stacked arrays, batch_size=2 |
| `provided_code/network_architectures.py` | XLA, float32 output layers |
| `provided_code/network_functions.py` | Reduced logging, .keras format, XLA on resume |

---

## Troubleshooting

### GPU not detected
```bash
pip install tensorflow[and-cuda]
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### OOM Error
- Reduce batch size: `batch_size: int = 1`
- Reduce filters: `--filters 32`

### XLA Compilation Slow
- Normal for first epoch (~3 min with batch_size=4)
- Subsequent epochs are faster
- Happens again after model resume

### Low GPU Utilization
- **This is expected for 3D convolutions**
- Memory bandwidth is the bottleneck, not compute
- Smaller batch size (2) is actually faster than larger (4)
- Use A100/H100 for higher throughput

### Disconnected from Terminal
- Training continues in background
- Check progress: `ls /workspace/results/*/models/`
- Check if running: `ps aux | grep python`

### Stacking Data Takes Too Long
- First run loads 200 patients into ~40GB RAM
- Normal time: ~30-60 seconds
- If >5 minutes: RunPod storage issue, try Ctrl+C and restart

---

## Future Improvements

1. **Loss logging to file** - Save epoch losses to CSV for monitoring
2. **TensorBoard integration** - Real-time training visualization
3. **Gradient checkpointing** - Trade compute for memory on larger models
4. **Multi-GPU training** - For A100 x2 or x4 setups
