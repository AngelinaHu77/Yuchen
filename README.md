# Event-Driven Stock Trend Prediction via BERT and ALBERT


## Methodology

1. Data Collection and Pre-Processing

    - The data has been pre-processed (steps can be seen in Pre-processing.ipynb) and saved in reuters_news_concatenated.pkl, which can be directly used later. (If you want to check, Rawdata file has all of the data we needed)
  
2. Models

    2.1 1D-CNN and BIDIR-GRU (Baselines.ipynb)
    
        - Vectorize news into a 2D integer tensor and pad sequences

        - Cluster labels and implement one-hot encoding

        - Data Split

        - Prepare Embedding Matrix using GloVe

        - Declare Architectues

        - Train, evaluate and test models (visualisation)
    
    2.2 Fine-tuning BERT/ALBERT (FinancePredictBert.py/FinancePredictALBert.py)
    
        - Load data for BERT/ALBERT and cluster labels

        - Tokenize and encode text data

        - Data Split

        - Download Pre-Trained Model (bert-base-uncased file/ALbert file)

        - Configure fine-tuning parameters

        - Train, evaluate using metrics and test models (visualisation in plotfig.py)


## Requirement
* Python 3
* Pytorch
* tensorflow-gpu 2.2.0
* pandas
* numpy
* requests
* tqdm
* multiprocessing
* string
* [NLTK](https://www.nltk.org/install.html)
* pickle
* keras 2.4.3
* keras-bert 0.88.0
* sklearn
* wget
* zipfile


