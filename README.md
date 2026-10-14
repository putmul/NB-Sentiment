This project contains Streamlit applications for English and Indonesian sentiment analysis.

## Features

- English sentiment analysis using TextBlob and VADER
- Indonesian sentiment analysis using a Naive Bayes classifier
- Indonesian text preprocessing with stemming and stopword removal
- Sentiment datasets in CSV format

## Project Files

- `bayes_en.py` — English sentiment analysis app
- `bayes_id.py` — Indonesian sentiment analysis app
- `preprocess.py` — Downloads and preprocesses the Indonesian dataset
- `preprocessed_Indonesian_Sentiment_Twitter_Dataset_Labeled.csv` — Dataset
- `preprocessed_Indonesian_Sentiment_Twitter_Dataset_Labeled_2000.csv` — Processed dataset

## Requirements

- Python 3.10 or newer
- Internet connection for installing packages
- Internet connection when running `preprocess.py`

## Installation

#### Clone the repository:

```bash
git clone https://github.com/putmul/NB-Sentiment/.git
cd NB-Sentiment
```

#### Create a virtual environment
Windows:
```bash
py -m venv .venv
.venv\Scripts\activate
```
macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Install the dependencies:
```bash
python -m pip install -r requirements.txt
```

#### Running the Applications
Run the Indonesian sentiment app: 
```bash
streamlit run bayes_id.py
```
Run the English sentiment app: 
```bash
streamlit run bayes_en.py
```

#### Preprocessing the Dataset
To download and regenerate the processed Indonesian dataset: 
```bash
python preprocess.py
```
The preprocessing script uses a public Indonesian Twitter sentiment dataset.
