# Dual-Region Network (DRN)
*A PyTorch implementation for odor discrimination from local field potentials (LFPs)*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT%20or%20Apache--2.0-green.svg)](LICENSE)
[![Cite](https://img.shields.io/badge/Cite-CITATION.cff-lightgrey.svg)](CITATION.cff)

---

## Overview

This repository contains the official implementation of the **Dual-Region Network (DRN)** introduced in:

> **Dual-Region Deep Learning for Odor Discrimination from Local Field Potentials**  
> *Zareh et al., 2025*

The DRN jointly analyzes neural activity from the **olfactory bulb (OB)** and **piriform cortex (PCx)** to classify odors. It features:

- **Region-Specific Networks (RSNs)** with Channel Attention (CA), a Conv block, Residual blocks, Spatial Attention (SA), and Feature Projection (FP)
- **Fusion Attention (FA)** that adaptively weighs OB vs. PCx features before classification
- A **classifier** (deep path + skip connection) for stable training
- Reproducible training with **focal loss**, **robust scaling**, **Mixup**, **5-fold CV**, and **logits-level ensembling**

> **Repo tagline:** PyTorch implementation of the Dual-Region Network (DRN) for odor classification from brain LFP signals. Includes region-specific networks, fusion attention, and a reproducible training pipeline.
