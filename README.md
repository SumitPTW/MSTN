# MSTN: Fast and Efficient Multivariate Time Series Model
Setup

# Install dependencies
- pip install torch numpy pandas scikit-learn
- pip install torch torchvision torchaudio
- pip install aeon
- pip install datasetsforecast aeon scikit-learn numpy pandas
- pip install scikit-learn numpy

# Installation - Clone repository
git clone https://github.com/SumitPTW/MSTN.git
cd MSTN


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

# Reproducing Experiments

# Forecasting (9 datasets)
  ETTh1, ETTh2, ETTm1, ETTm2, ECL, Weather, Traffic, Exchange, ILI

# Classification (10 datasets)
  EthanolConcentration, FaceDetection, Handwriting, Heartbeat, JapaneseVowels, PEMS-SF, SelfRegulationSCP1, SelfRegulationSCP2, SpokenArabicDigits, UWaveGestureLibrary

# Imputation (6 datasets)
  ETTh1, ETTh2, ETTm1, ETTm2, ECL, Weather


# Model Architecture
MSTN implements a Multi-scale Temporal Network with Early Temporal Aggregation (ETA):
<img width="750" height="700" alt="MSTN FINAL" src="https://github.com/user-attachments/assets/d605f7e2-8c21-409e-9cbc-20d5d4ef22e4" />

# MSTN multi-scale signal processing pipeline
<img width="750" height="700" alt="image" src="https://github.com/user-attachments/assets/63cef800-2781-49dc-99d6-cd6bc443c832" />




# Key Innovations:

Early Temporal Aggregation (ETA): Collapses sequence dimension early for O(1) inference

Dual-path Encoding: CNN (local) + Transformer/BiLSTM (global)

Parameter Efficiency: ~1.04M (Transformer) and ~0.40M (BiLSTM) parameters

📁 Project Structure
text
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


bash
Configuration
Model: MSTN-Transformer or MSTN-BiLSTM

Lookback window: 96 (36 for ILI)

Prediction horizons: 96, 192, 336, 720 (24, 36, 48, 60 for ILI)

Imputation mask ratios: 12.5%, 25%, 37.5%, 50%

Evaluation metrics: MSE, MAE (forecasting/imputation), Accuracy (classification)

bash
#Acknowledgement

All experiment datasets are public:

Long-term Forecasting and Imputation: Autoformer

Classification: UEA Archive

Cross-Domain Datasets: See references in paper (UCI-HAR, PAMAP2, Rodegast, Boubezoul, ActBeCalf, MetroPT3, NASA)



## Verification
Run `python test_mstn.py` to confirm:
- MSTN-Transformer: 1,038,818 parameters (~1.04M)
- MSTN-BiLSTM: 381,410 parameters (~0.40M)
- Both models compile and execute forward passes
  
