#MSTN: Fast and Efficient Multivariate Time Series Model
Setup


# Install dependencies
pip install torch torchvision torchaudio
pip install datasetsforecast aeon scikit-learn numpy pandas
Reproducing Experiments
Forecasting (9 datasets)
bash
Installation
bash
# Clone repository
git clone https://github.com/SumitPTW/MSTN.git
cd MSTN

# Install dependencies
pip install torch numpy pandas scikit-learn
Quick Verification for Reviewers
Run this simple test to verify MSTN works:

bash
python test_mstn.py
Expected Output:

text
✅ MSTN_Transformer created successfully
   Total parameters: 1,038,818 (~1.04M in paper)
✅ Forward pass successful

✅ MSTN_BiLSTM created successfully
   Total parameters: 381,410 (~0.40M in paper)
✅ Forward pass successful

✅ Tests passed! MSTN forward passes.
📋 Configuration
Setting	Value
Model	MSTN-Transformer or MSTN-BiLSTM
Lookback window	96 (36 for ILI)
Prediction horizons	96, 192, 336, 720 (24, 36, 48, 60 for ILI)
Imputation mask ratios	12.5%, 25%, 37.5%, 50%
Evaluation metrics	MSE, MAE (forecasting/imputation), Accuracy (classification)
🔬 Reproducing Experiments
Forecasting (9 datasets)
bash
# Example: ETTh1 forecasting with MSTN-Transformer
python run_main.py --task_name long_term_forecast \
  --model MSTN_Transformer \
  --data ETTh1 \
  --seq_len 96 \
  --pred_len 96
Available datasets: ETTh1, ETTh2, ETTm1, ETTm2, ECL, Weather, Traffic, Exchange, ILI

Classification - UEA Archive (10 datasets)
bash
# Example: Heartbeat classification
python run_main.py --task_name classification \
  --model MSTN_Transformer \
  --data Heartbeat
Available datasets: EthanolConcentration, FaceDetection, Handwriting, Heartbeat, JapaneseVowels, PEMS-SF, SelfRegulationSCP1, SelfRegulationSCP2, SpokenArabicDigits, UWaveGestureLibrary

Imputation (6 datasets)
bash
# Example: ETTh1 imputation with 25% missing
python run_main.py --task_name imputation \
  --model MSTN_Transformer \
  --data ETTh1 \
  --mask_rate 0.25
📁 Dataset Preparation
Forecasting Datasets
bash
# Create dataset directory
mkdir -p ./dataset

# Download ETT datasets (example for ETTh1)
wget -O ./dataset/ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv

# Other ETT datasets (similarly):
# ETTh2: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv
# ETTm1: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv
# ETTm2: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv
Other forecasting datasets (download from original sources cited in paper):

ECL: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

Weather: https://www.bgc-jena.mpg.de/wetter/

Traffic: https://pems.dot.ca.gov

Exchange: https://github.com/laiguokun/multivariate-time-series-data

ILI: https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html

UEA Classification Datasets: Automatically handled by data loader.

📊 Model Architecture
MSTN implements a Multi-scale Temporal Network with Early Temporal Aggregation (ETA):


┌─────────────────────────────────────────────────────┐
│ Input: [Batch, Sequence Length, Channels]           │
├─────────────────────────────────────────────────────┤
│ Dual-path Encoding:                                 │
│ • CNN Branch: Multi-scale local patterns            │
│ • Sequence Branch: Transformer/BiLSTM (long-range)  │
├─────────────────────────────────────────────────────┤
│ Early Temporal Aggregation (ETA): L → 1             │
├─────────────────────────────────────────────────────┤
│ Feature Concatenation                               │
├─────────────────────────────────────────────────────┤
│ Self-Gated Fusion + SE + MHA Refinement             │
├─────────────────────────────────────────────────────┤
│ Task-specific Output                                │
└─────────────────────────────────────────────────────┘

Key Innovations:

Early Temporal Aggregation (ETA): Collapses sequence dimension early for O(1) inference

Dual-path Encoding: CNN (local) + Transformer/BiLSTM (global)

Parameter Efficiency: ~1.04M (Transformer) and ~0.40M (BiLSTM) parameters

📁 Project Structure
text
MSTN/
├── models/
│   ├── MSTN_Transformer.py    # MSTN-Transformer implementation
│   ├── MSTN_BiLSTM.py         # MSTN-BiLSTM implementation
│   └── __init__.py
├── layers/
│   ├── eta_module.py          # Early Temporal Aggregation
│   ├── sgf_module.py          # Self-Gated Fusion
│   ├── se_block.py            # Squeeze-and-Excitation
│   ├── mha_recalibration.py   # Multi-head Attention recalibration
│   └── mstn_modules.py        # CNN/BiLSTM/Transformer pathways
├── data_provider/
│   ├── data_factory.py        # Dataset loader
│   └── data_loader.py
├── utils/
│   ├── tools.py
│   └── timefeatures.py
├── run_main.py                # Main training/evaluation script
├── test_mstn.py               # Verification test for reviewers
├── requirements.txt           # Dependencies
└── README.md                  # This file


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
  
