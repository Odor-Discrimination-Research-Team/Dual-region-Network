# Dual-Region Network (DRN)
*A PyTorch implementation for odor discrimination from local field potentials (LFPs)*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT%20or%20Apache--2.0-green.svg)](LICENSE)
[![Cite](https://img.shields.io/badge/Cite-CITATION.cff-lightgrey.svg)](CITATION.cff)

---

Overview
--------
The DRN is designed to classify odor-evoked local field potentials (LFPs) recorded
from two brain regions: the olfactory bulb (OB) and the piriform cortex (PCx).
It uses parallel Region-Specific Networks (RSNs) to process each region
independently, then fuses them with an attention mechanism before classification.

Main Components
---------------
1. **ChannelAttention** (CA)
   - Learns channel-wise weights using distinct MLPs for max and average pooled
     features.

2. **SpatialAttention** (SA)
   - Learns spatial importance using pooled features, convolution, and sigmoid
     gating.

3. **ResidualBlock** (RB)
   - Pre-activation residual block with channel attention; supports downsampling
     by stride=2.

4. **RegionSpecificNet** (RSN)
   - Full processing pipeline per region:
     CA → ConvBlock → MaxPool → N×RB → SA → Global pooling → Feature Projection.

5. **DualRegionNet** (DRN)
   - Combines two RSNs (OB and PCx), applies Fusion Attention (FA), and feeds
     fused features into:
       • A deep classifier block
       • A shallow skip path
     Logits from both are summed for final predictions.

Training Utilities
------------------
- **DualInputDataset**: wraps OB, PCx, and labels into a PyTorch dataset.
- **FocalLoss**: γ=2.0, reduction='sum', robust to class imbalance.
- **Mixup**: on-the-fly augmentation applied consistently across OB/PCx.
- **train_epoch / validate**: single-epoch training and validation loops.
- **train_model**: full training routine (AdamW, gradient clipping, 70 epochs by default).
- **train_ensemble**: trains with K-fold CV and assembles an ensemble prediction
  on a held-out balanced test set (50 samples/class).

Usage
-----
Typical workflow:
    • Prepare OB and PCx feature arrays shaped (N, Channels, Bands).
    • Encode class labels to integers.
    • Call `train_ensemble(...)` with data arrays, config dict, and LabelEncoder.

Dependencies
------------
- Python 3.10+
- PyTorch >= 2.0
- NumPy, SciPy
- scikit-learn
- tqdm
