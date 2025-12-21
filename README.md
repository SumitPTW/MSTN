# MSTN: Fast and Efficient Multivariate Time Series Model
Multi-scale Temporal Network (MSTN), a hybrid neural architecture grounded in an Early Temporal Aggregation principle. MSTN integrates three complementary components: (i) a multi-scale convolutional encoder that captures fine-grained local structure; (ii) a sequence modeling module that learns long-range dependencies through either recurrent or attention-based mechanisms; and (iii) a self-gated fusion stage incorporating squeeze–excitation and multi-head attention to dynamically modulate cross-scale representations. This design enables MSTN to flexibly model temporal patterns spanning milliseconds to extended horizons, while avoiding the computational burden typically associated with long-context models. Across extensive benchmarks covering forecasting, imputation, classification, and cross-dataset generalization, MSTN achieves state-of-the-art performance, establishing new best results on 24 of 32 datasets, while remaining lightweight (≈1M params) and suitable for low-latency (<1 sec, often in milliseconds), resource-constrained deployment.

# Model Architecture
<img width="750" height="700" alt="MSTN FINAL" src="https://github.com/user-attachments/assets/d605f7e2-8c21-409e-9cbc-20d5d4ef22e4" />

# MSTN multi-scale signal processing pipeline
<img width="750" height="700" alt="image" src="https://github.com/user-attachments/assets/82936e1d-4959-47dc-bd9e-5864e7a17ad2" />

# Core Innovation: Dual-Path Design and Early Temporal Aggregation (ETA)

Dual-path Encoding: CNN (local) + Transformer/BiLSTM (global): parallel encoding strategy in which a global sequence-modeling pathway (Transformer or BiLSTM; CD strategy) captures long-range temporal dependencies, while a lightweight convolutional pathway (CI strategy; O(L)) extracts fine-grained local temporal patterns.

Efficiency through ETA: collapsing O(L2) to O(1). MSTN applies the ETA mechanism immediately after the high-capacity encoders, collapsing the temporal dimension (L → 1) via learned sequence aggregation. By performing the computationally intensive operations (e.g., O(L2) Transformer self-attention) before this aggregation step, the subsequent refinement layers—including feature fusion, SE recalibration, MHA, and prediction modules—operate with a fixed O(1) cost with respect to L. 

Parameter Efficiency: ~1.04M (Transformer) and ~0.40M (BiLSTM) parameters

# Project Structure

MSTN/
├── models/
│ ├── MSTN_Transformer.py # MSTN-Transformer implementation
│ ├── MSTN_BiLSTM.py # MSTN-BiLSTM implementation
│ └── init.py
├── layers/
│ ├── eta_module.py # Early Temporal Aggregation
│ ├── sgf_module.py # Self-Gated Fusion
│ ├── se_block.py # Squeeze-and-Excitation
│ ├── mha_recalibration.py # Multi-head Attention recalibration
│ └── mstn_modules.py # CNN/BiLSTM/Transformer pathways
├── data_provider/
│ ├── data_factory.py # Dataset loader
│ └── data_loader.py
├── utils/
│ ├── tools.py
│ └── timefeatures.py
├── run_main.py # Main training/evaluation script
├── test_mstn.py # Verification test for reviewers
├── requirements.txt # Dependencies
└── README.md # This file


# Install dependencies
- pip install torch numpy pandas scikit-learn
- pip install torch torchvision torchaudio
- pip install aeon
- pip install datasetsforecast aeon scikit-learn numpy pandas
- pip install scikit-learn numpy

# Installation - Clone repository
- git clone https://github.com/SumitPTW/MSTN.git
- cd MSTN


# Quick Verification- Run this simple test to verify MSTN works:

Run python test_mstn.py

Expected Output:
- MSTN_Transformer created successfully
- Total parameters: 1,038,818
- Forward pass successful
- MSTN_BiLSTM created successfully
- Total parameters: 381,410
- Total parameters: 381,410
- Tests passed! MSTN forward passes.

  
# Configuration
- Setting	Value
- Model	MSTN-Transformer or MSTN-BiLSTM
- Lookback window	96 (36 for ILI)
- Prediction horizons	96, 192, 336, 720 (24, 36, 48, 60 for ILI)
- Imputation mask ratios	12.5%, 25%, 37.5%, 50%
- Evaluation metrics	MSE, MAE (forecasting/imputation), Accuracy (classification)

# Reproducing Experiments - All experiment datasets are public (Links / References given in paper)

# Forecasting (9 datasets)
  ETTh1, ETTh2, ETTm1, ETTm2, ECL, Weather, Traffic, Exchange, ILI

# Classification (10 datasets)
  EthanolConcentration, FaceDetection, Handwriting, Heartbeat, JapaneseVowels, PEMS-SF, SelfRegulationSCP1, SelfRegulationSCP2, SpokenArabicDigits, UWaveGestureLibrary

# Imputation (6 datasets)
  ETTh1, ETTh2, ETTm1, ETTm2, ECL, Weather
  
# Cross-Domain Generalizability (7 datasets)
  UCI-HAR, PAMAP2, Rodegast, Boubezoul, ActBeCalf, MetroPT3, NASA





# Verification
Run `python test_mstn.py` to confirm:
- MSTN-Transformer: 1,038,818 parameters (~1.04M)
- MSTN-BiLSTM: 381,410 parameters (~0.40M)
- Both models compile and execute forward passes
  
