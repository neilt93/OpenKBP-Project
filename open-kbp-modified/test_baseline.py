#!/usr/bin/env python3
"""
Test baseline configuration to match original Colab results.

This runs with settings matching the original OpenKBP code:
- No normalization (original has none)
- No SE blocks (original has none)
- No DVH loss (original has none)
- No augmentation (original has none)
- No mixed precision (original uses float32)
- No XLA JIT (original doesn't use it)

Usage:
    python test_baseline.py --filters 64 --epochs 100
"""
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Test baseline (original Colab) configuration')
    parser.add_argument('--filters', type=int, default=64, help='Number of filters')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    args = parser.parse_args()

    # Run with original baseline settings - disable ALL our modifications
    cmd = [
        sys.executable, 'runpod_train.py',
        '--filters', str(args.filters),
        '--epochs', str(args.epochs),
        '--no-normalize',       # Original has no normalization
        '--no-mixed-precision', # Original uses float32
        '--no-jit',             # Original has no XLA JIT
        '--no-cache',           # Original has no data caching
        # No --use-se (original has no SE blocks)
        # No --use-dvh (original has no DVH loss)
        # No --use-aug (original has no augmentation)
    ]

    print("=" * 60)
    print("Running BASELINE configuration (matches original Colab)")
    print("=" * 60)
    print("Settings:")
    print("  - Normalization: OFF")
    print("  - Mixed precision: OFF (float32)")
    print("  - XLA JIT: OFF")
    print("  - Data caching: OFF")
    print("  - SE blocks: OFF")
    print("  - DVH loss: OFF")
    print("  - Augmentation: OFF")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    subprocess.run(cmd)

if __name__ == "__main__":
    main()
