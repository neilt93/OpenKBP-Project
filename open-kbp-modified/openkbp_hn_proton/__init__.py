"""OpenKBP H&N proton dose-prediction pipeline.

Phase 1: bridge OpenKBP photon-dataset anatomy into matRad to generate proton
IMPT ground-truth dose, then train the existing 3D U-Net to predict proton dose.

This package is intentionally self-contained (it does not import the photon
robustness module) so the bridge can be tested without the full dataset.
"""
