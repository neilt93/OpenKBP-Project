#!/bin/bash
#
# Train multiple models with different seeds for ensemble prediction
#
# Usage: ./train_ensemble.sh [filters] [epochs] [num_models]
# Example: ./train_ensemble.sh 64 100 3
#

FILTERS=${1:-64}
EPOCHS=${2:-100}
NUM_MODELS=${3:-3}

# Seeds for reproducibility (can be customized)
SEEDS=(42 123 456 789 1337)

echo "========================================"
echo "Training ${NUM_MODELS} models for ensemble"
echo "Filters: ${FILTERS}, Epochs: ${EPOCHS}"
echo "========================================"

# Track trained model names
MODEL_NAMES=""

for i in $(seq 0 $((NUM_MODELS - 1))); do
    SEED=${SEEDS[$i]}
    echo ""
    echo "Training model $((i + 1))/${NUM_MODELS} with seed ${SEED}..."
    echo ""

    python runpod_train.py \
        --filters ${FILTERS} \
        --epochs ${EPOCHS} \
        --use-se \
        --use-aug \
        --seed ${SEED}

    # Build model name to match what runpod_train.py creates
    MODEL_NAME="${FILTERS}filter_${EPOCHS}epoch_SE_AUG_NORM_seed${SEED}"
    MODEL_NAMES="${MODEL_NAMES} ${MODEL_NAME}"
done

echo ""
echo "========================================"
echo "All models trained!"
echo "Model names:${MODEL_NAMES}"
echo "========================================"
echo ""
echo "To run ensemble prediction:"
echo "python ensemble_predict.py --models${MODEL_NAMES} --epoch ${EPOCHS} --output ensemble_${NUM_MODELS}models"
