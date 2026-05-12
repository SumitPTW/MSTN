# MSTN: Fast and Efficient Multivariate Time Series Prediction Model
Real-world time series often exhibit strong non-stationarity, complex nonlinear dynamics, and behavior expressed across multiple temporal scales, from rapid local fluctuations to
slow-evolving long-range trends. However, many contemporary architectures impose rigid, fixed-scale structural priors—such as patch-based tokenization, predefined receptive fields,
or frozen backbone encoders—which can over-regularize temporal dynamics and limit adaptability to abrupt high-magnitude events. To handle this, we introduce the Multi-scale Tem-
poral Network (MSTN), a hybrid neural architecture grounded in an Early Temporal Aggregation principle. MSTN integrates three complementary components: (i) a multi-scale
convolutional encoder that captures fine-grained local structure; (ii) a sequence modeling module that learns long-range dependencies through either recurrent or attention-based
mechanisms; and (iii) a self-gated fusion stage incorporating squeeze–excitation and a single dense layer to dynamically reweight and fuse multi-scale representations. 

# Model Architecture
<img width="3522" height="2136" alt="MSTN" src="https://github.com/user-attachments/assets/2ca0313e-600e-41e6-b03c-d251a144ae47" />

# MSTN multi-scale signal processing pipeline
<img width="3522" height="2136" alt="image" src="https://github.com/user-attachments/assets/82936e1d-4959-47dc-bd9e-5864e7a17ad2" />

# Core Innovation: Dual-Path Design and Early Temporal Aggregation (ETA)

Dual-path Encoding: CNN (local) + Transformer/BiLSTM (global): parallel encoding strategy in which a global sequence-modeling pathway (Transformer or BiLSTM; CD strategy) captures long-range temporal dependencies, while a lightweight convolutional pathway (CI strategy; O(L)) extracts fine-grained local temporal patterns.

Efficiency through ETA: collapsing O(L2) to O(1). MSTN applies the ETA mechanism immediately after the high-capacity encoders, collapsing the temporal dimension (L → 1) via learned sequence aggregation. By performing the computationally intensive operations (e.g., O(L2) Transformer self-attention) before this aggregation step, the subsequent refinement layers—including feature fusion, SE recalibration, MHA, and prediction modules—operate with a fixed O(1) cost with respect to L. 

Parameter Efficiency: ~1.04M (Transformer) and ~0.40M (BiLSTM) parameters

## Project Structure

```text
MSTN/
├── models/
│   ├── MSTN_Transformer.py   # MSTN-Transformer implementation
│   ├── MSTN_BiLSTM.py        # MSTN-BiLSTM implementation
│   └── __init__.py
├── layers/
│   ├── eta_module.py         # Early Temporal Aggregation
│   ├── sgf_module.py         # Self-Gated Fusion
│   ├── se_block.py           # Squeeze-and-Excitation
│   ├── single_dense_layer.py # Single Dense layer 
│   └── mstn_modules.py       # CNN/BiLSTM/Transformer pathways
├── data_provider/
│   ├── data_factory.py       # Dataset loader
│   └── data_loader.py
├── utils/
│   ├── tools.py
│   └── timefeatures.py
├── run_main.py               # Main training/evaluation script
├── test_mstn.py              # Verification test for reviewers
├── requirements.txt          # Dependencies
└── README.md                 # This file
```


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
- Tests passed! MSTN forward passes.

  
# Configuration
- Setting	Value
- Model	MSTN-Transformer or MSTN-BiLSTM
- Lookback window	96 
- Prediction horizons	12, 24, 48, 96
- Imputation mask ratios	12.5%, 25%, 37.5%, 50%
- Evaluation metrics	MSE, MAE (forecasting/imputation), Accuracy (classification)

# Reproducing Experiments - All experiment datasets are public (Links / References given in paper)

# Imputation (6 datasets)
  ETTh1, ETTh2, ETTm1, ETTm2, ECL, Weather
  
# Forecasting (4 PEMS datasets)
  PEMS03, PEMS04, PEMS07, PEMS08

# Classification (10 datasets)
  EthanolConcentration, FaceDetection, Handwriting, Heartbeat, JapaneseVowels, PEMS-SF, SelfRegulationSCP1, SelfRegulationSCP2, SpokenArabicDigits, UWaveGestureLibrary
  
# Cross-Dataset Generalizability (7 datasets)
  UCI-HAR, PAMAP2, Rodegast, Boubezoul, ActBeCalf, MetroPT3, NASA


