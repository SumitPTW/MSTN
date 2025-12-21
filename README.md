MSTN: Fast and Efficient Multivariate Time Series Model
Setup
bash
# Clone repository
git clone https://github.com/yourusername/MSTN.git
cd MSTN

# Install dependencies
pip install torch torchvision torchaudio
pip install datasetsforecast aeon scikit-learn numpy pandas
Reproducing Experiments
Forecasting (9 datasets)
bash
# Example: ETTm2 forecasting
python run_main.py --task forecasting --dataset ETTm2 --seq_len 96 --pred_len 96

# All forecasting datasets: ETTh1, ETTh2, ETTm1, ETTm2, ECL, Weather, Traffic, Exchange, ILI
Classification - UEA Archive (10 datasets)
bash
# Example: Heartbeat classification
python run_main.py --task classification --dataset Heartbeat

# All UEA datasets: EthanolConcentration, FaceDetection, Handwriting, Heartbeat, 
# JapaneseVowels, PEMS-SF, SelfRegulationSCP1, SelfRegulationSCP2, 
# SpokenArabicDigits, UWaveGestureLibrary
Imputation (6 datasets)
bash
# Example: ETTh1 imputation with 25% missing
python run_main.py --task imputation --dataset ETTh1 --mask_ratio 0.25
Cross-Domain Generalization (7 datasets)
bash
# Example: UCI-HAR cross-domain classification
python run_main.py --task classification --dataset UCI-HAR
Dataset Preparation
Forecasting Datasets
bash
# ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2)
mkdir -p ./datasets/forecasting/ETTh1 && wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv -O ./datasets/forecasting/ETTh1/data.csv
mkdir -p ./datasets/forecasting/ETTh2 && wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv -O ./datasets/forecasting/ETTh2/data.csv
mkdir -p ./datasets/forecasting/ETTm1 && wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv -O ./datasets/forecasting/ETTm1/data.csv
mkdir -p ./datasets/forecasting/ETTm2 && wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv -O ./datasets/forecasting/ETTm2/data.csv

# Other forecasting datasets: Download from original sources cited in paper
# ECL: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
# Weather: https://www.bgc-jena.mpg.de/wetter/
# Traffic: https://pems.dot.ca.gov
# Exchange: https://github.com/laiguokun/multivariate-time-series-data
# ILI: https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html
UEA Classification Datasets
Automatically handled by aeon package. No manual download required.

Cross-Domain Datasets
Download from original sources:

UCI-HAR: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

PAMAP2: https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring

Rodegast: DOI: 10.18419/darus-3301

Boubezoul: Reference in paper

ActBeCalf: Reference in paper

MetroPT3: https://archive.ics.uci.edu/dataset/751/metropt+3

NASA: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

Place each dataset in: ./datasets/cross_domain/{dataset_name}/data.csv

Configuration
Model: MSTN-Transformer or MSTN-BiLSTM

Lookback window: 96 (36 for ILI)

Prediction horizons: 96, 192, 336, 720 (24, 36, 48, 60 for ILI)

Imputation mask ratios: 12.5%, 25%, 37.5%, 50%

Evaluation metrics: MSE, MAE (forecasting/imputation), Accuracy (classification)




# MSTN: Fast and Efficient Multivariate Time Series Model

## Setup

Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install datasetsforecast aeon scikit-learn numpy pandas
Reproducing Experiments
Forecasting (9 datasets)
bash
# Example: ETTm2 forecasting
python run_main.py --task forecasting --dataset ETTm2 --seq_len 96 --pred_len 96

# All forecasting datasets: ETTh1, ETTh2, ETTm1, ETTm2, ECL, Weather, Traffic, Exchange, ILI
Classification - UEA Archive (10 datasets)
bash
# Example: Heartbeat classification
python run_main.py --task classification --dataset Heartbeat

# All UEA datasets: EthanolConcentration, FaceDetection, Handwriting, Heartbeat,
# JapaneseVowels, PEMS-SF, SelfRegulationSCP1, SelfRegulationSCP2,
# SpokenArabicDigits, UWaveGestureLibrary
Imputation (6 datasets)
bash
# Example: ETTh1 imputation with 25% missing
python run_main.py --task imputation --dataset ETTh1 --mask_ratio 0.25
Cross-Domain Generalization (7 datasets)
bash
# Example: UCI-HAR cross-domain classification
python run_main.py --task classification --dataset UCI-HAR
Dataset Preparation
Forecasting Datasets
bash
# ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2)
mkdir -p ./datasets/forecasting/ETTh1 && wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv -O ./datasets/forecasting/ETTh1/data.csv
mkdir -p ./datasets/forecasting/ETTh2 && wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv -O ./datasets/forecasting/ETTh2/data.csv
mkdir -p ./datasets/forecasting/ETTm1 && wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv -O ./datasets/forecasting/ETTm1/data.csv
mkdir -p ./datasets/forecasting/ETTm2 && wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv -O ./datasets/forecasting/ETTm2/data.csv

# Other forecasting datasets: Download from original sources cited in paper
# ECL: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
# Weather: https://www.bgc-jena.mpg.de/wetter/
# Traffic: https://pems.dot.ca.gov
# Exchange: https://github.com/laiguokun/multivariate-time-series-data
# ILI: https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html

UEA Classification Datasets
Automatically handled by aeon package. No manual download required.

#Cross-Domain Datasets (Link/Reference in paper)

UCI-HAR, PAMAP2, Rodegast, Boubezoul, ActBeCalf, MetroPT3, NASA, 

Configuration
Model: MSTN-Transformer or MSTN-BiLSTM

Lookback window: 96 (36 for ILI)

Prediction horizons: 96, 192, 336, 720 (24, 36, 48, 60 for ILI)

Imputation mask ratios: 12.5%, 25%, 37.5%, 50%

Evaluation metrics: MSE, MAE (forecasting/imputation), Accuracy (classification)
