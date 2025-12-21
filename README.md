# MSTN
Private repository containing the source code for the Multi-scale Temporal Network (MSTN)



## Dataset Setup

All datasets used in this work are publicly available. Due to their collective size (~15GB), they are not included in this repository. To reproduce our experiments, please download them using the links below, which correspond to the citations in our paper.

### 1. Create Directory Structure
First, create the expected folder structure in your local copy of the repository:
```bash
mkdir -p datasets/{forecasting,UEA,cross_domain}

2. Download Forecasting & Imputation Datasets
These include ETT, ECL, Weather, Traffic, Exchange, and ILI.

Source: Follow the official download instructions from the Autoformer repository.

Action: Download the .csv files (e.g., ETTh1.csv, ECL.csv) and place them in the datasets/forecasting/ folder.

3. Download UEA Classification Datasets
These are the 10 multivariate datasets from the UEA archive.

Source: Download from the official UEA & UCR Time Series Classification Repository.

Action: For each dataset (e.g., EthanolConcentration), extract the _TRAIN.ts and _TEST.ts files into a dedicated folder: datasets/UEA/EthanolConcentration/.

4. Download Cross-Domain Datasets
These include PAMAP2, UCI-HAR, NASA Turbofan, and others from Table 3b.

PAMAP2: Download from the UCI Machine Learning Repository.

NASA Turbofan: Download from the NASA Prognostics Center.

Other datasets: Refer to the individual citations in the "Benchmark Datasets" section of our paper.

Action: Place each dataset in its own subfolder under datasets/cross_domain/ (e.g., datasets/cross_domain/PAMAP2/).

5. Expected Final Structure
After downloading, your datasets folder should look like this:

text
datasets/
├── forecasting/               # For .csv files (ETT, ECL, Weather, etc.)
│   ├── ETTh1.csv
│   ├── ECL.csv
│   └── ...
├── UEA/                      # For UEA archive .ts files
│   ├── EthanolConcentration/
│   │   ├── EthanolConcentration_TRAIN.ts
│   │   └── EthanolConcentration_TEST.ts
│   ├── FaceDetection/
│   └── ... (8 more datasets)
└── cross_domain/             # For PAMAP2, NASA, etc.
    ├── PAMAP2/               # Contains PAMAP2 data files
    ├── UCI_HAR/              # Contains UCI-HAR data files
    └── ... (other datasets)
6. Running Experiments
Set the --root_path argument in your command to point to the correct parent folder.

Example for forecasting:

bash
python run_main.py --task_name forecasting --dataset_name ETTh1 --root_path ./datasets/forecasting/ --data_path ETTh1.csv
Example for UEA classification:

bash
python run_main.py --task_name classification --dataset_name EthanolConcentration --root_path ./datasets/UEA/
Example for cross-domain (PAMAP2):

bash
python run_main.py --task_name classification --dataset_name PAMAP2 --root_path ./datasets/cross_domain/
