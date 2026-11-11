# Sentiment-Analysis-with-huggingface_transformers
# Sentiment Analysis with Hugging Face Transformers

Fine-tuning a pretrained transformer (DistilBERT by default) to classify text as **Positive**, **Negative** or **Neutral**, using a dataset downloaded from Kaggle.

## Overview

This project takes a labelled sentiment dataset (CSV), cleans it, and fine-tunes a Hugging Face transformer using the `Trainer` API. The trained model is saved and can be used for inference with a single `pipeline` call.

**Pipeline**

1. Load the Kaggle CSV with pandas
2. Detect (or use the configured) text and label columns
3. Clean the data: drop missing values, empty texts and duplicates, normalise label spelling
4. Encode labels as integers and split into train/test (80/20, stratified)
5. Tokenize with the model's tokenizer (dynamic padding)
6. Fine-tune with `Trainer`, evaluating accuracy and macro F1 each epoch
7. Save the best model and run sample predictions

## Project structure

```
.
├── finetune_kaggle.py     # main training script
├── data.csv               # your Kaggle dataset (not included)
├── sentiment-model/       # saved model and tokenizer (created after training)
└── README.md
```

## Requirements

- Python 3.9+
- A GPU is recommended (Google Colab's free T4 is enough)

```bash
pip install transformers datasets evaluate accelerate scikit-learn torch pandas
```

## Dataset

Download a sentiment dataset from [Kaggle](https://www.kaggle.com/datasets) and place the CSV in the project folder (or upload it to Colab). The file needs:

| Column type | Description |
|-------------|-------------|
| Text        | The review, tweet or sentence to classify |
| Label       | `Positive`, `Negative` or `Neutral` (any capitalisation) |

Column names can be anything. The script detects them automatically, or you can set them manually.

## Usage

Open `finetune_kaggle.py` and edit the **SETTINGS** block:

```python
CSV_PATH = "/content/data.csv"   # path to your CSV
TEXT_COL = None                  # column with the text (None = auto-detect)
LABEL_COL = None                 # column with the sentiment (None = auto-detect)
MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
SAMPLE_SIZE = None               # e.g. 5000 for a quick test run
```

Then run it:

```bash
python finetune_kaggle.py
```

In Google Colab, choose **Runtime → Change runtime type → T4 GPU**, uncomment `fp16=True` in `TrainingArguments`, then paste the script into a cell (or run `!python finetune_kaggle.py`).

> Tip: start with `SAMPLE_SIZE = 5000` to confirm everything works, then set it back to `None` to train on the full dataset.

## Choosing a model

| Model | Best for |
|-------|----------|
| `distilbert-base-uncased` | English text, fast and lightweight (default) |
| `bert-base-uncased` | English text, slightly higher accuracy, slower |
| `xlm-roberta-base` | Multilingual text, including Swahili |

## Results

Fill in after training:

| Metric   | Value |
|----------|-------|
| Accuracy | _TBD_ |
| F1 (macro) | _TBD_ |

## Using the trained model

```python
from transformers import pipeline

clf = pipeline("text-classification", model="sentiment-model")

print(clf("I love this product, it works great!"))
# [{'label': 'Positive', 'score': 0.98}]
```

## Troubleshooting

- **`NaN` labels after mapping:** your labels contain values other than Positive/Negative/Neutral. Check them with `df["label"].value_counts(dropna=False)`.
- **`UnicodeDecodeError` when reading the CSV:** the script retries with `latin-1` automatically.
- **Out of memory on GPU:** reduce `BATCH_SIZE` (e.g. 8) or `MAX_LENGTH` (e.g. 128).
- **Training is very slow:** make sure a GPU is enabled and use `SAMPLE_SIZE` while testing.

## License

Specify your license here (e.g. MIT). Check the Kaggle dataset's own license before redistributing the data.
