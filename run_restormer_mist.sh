#!/bin/bash

# ============================================================
# Run Restormer on the full MIST dataset
# Tasks:
#   - Gaussian Color Denoising (blind)
#   - Real Denoising (SIDD)
#
# Stains:
#   ER, HER2, Ki67, PR
#
# Splits:
#   trainA, trainB, valA, valB
#
# Usage:
#   bash run_restormer_mist.sh
# ============================================================

BASE_DATA="../../data/MIST"
STAINS=("ER" "HER2" "Ki67" "PR")
SPLITS=("trainA" "trainB" "valA" "valB")

echo "====================================================="
echo " Running Restormer on MIST dataset"
echo "====================================================="

for STAIN in "${STAINS[@]}"; do
    echo ""
    echo "-----------------------------------------------------"
    echo " Processing stain: $STAIN"
    echo "-----------------------------------------------------"

    for SPLIT in "${SPLITS[@]}"; do
        INPUT_DIR="$BASE_DATA/$STAIN/TrainValAB/$SPLIT"

        # Gaussian Color Denoising
        OUT_GAUSS="results/MIST_${STAIN}_Gaussian/${SPLIT}"
        echo "  [Gaussian] $STAIN - $SPLIT"
        python demo.py \
            --task Gaussian_Color_Denoising \
            --input_dir "$INPUT_DIR" \
            --result_dir "$OUT_GAUSS"

        # Real Denoising (SIDD)
        OUT_REAL="results/MIST_${STAIN}_Real/${SPLIT}"
        echo "  [Real]     $STAIN - $SPLIT"
        python demo.py \
            --task Real_Denoising \
            --input_dir "$INPUT_DIR" \
            --result_dir "$OUT_REAL"

    done
done

echo ""
echo "====================================================="
echo " All Restormer inference on MIST completed."
echo " Results saved under: Restormer/results/"
echo "====================================================="
