# Sentiment Analysis of Amazon Fine Food Reviews

A supervised, multi-class NLP project that classifies raw product reviews as **Positive**, **Neutral**, or **Negative** — without a human reading them first.

## Problem

The [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset provides a 1–5 star rating rather than a sentiment label, so the label mapping was defined as part of the problem:

| Stars | Sentiment |
|---|---|
| 4 – 5 | Positive |
| 3 | Neutral |
| 1 – 2 | Negative |

## Dataset

- **568,454** raw reviews
- **568,399** remaining after removing rows with missing values and correcting two rows with an impossible helpfulness ratio (99.99% retained)
- **113,680** reviews held out as the test set

Key EDA findings:
- Score distribution is heavily skewed toward 5 stars, which carries directly into class imbalance for the sentiment labels — the central modeling challenge.
- 1-star and 5-star reviews attract more helpfulness votes than 3–4 star reviews (a U-shaped pattern), suggesting readers engage more with strongly opinionated reviews.
- Helpfulness ratio rises with star rating.

## Methodology

1. **Clean** — strip HTML and punctuation from raw review text
2. **Lemmatize** — reduce words to root form and remove stopwords
3. **Combine fields** — merge cleaned review body with the review summary
4. **Train/test split** — performed *before* any class balancing
5. **Oversample** — minority sentiment classes in the training set only, oversampled to match the majority class size (test set is left untouched to avoid leakage)

Two models were trained on identical training data and evaluated on the same held-out test set:

| Model | Pipeline |
|---|---|
| **Naive Bayes** | TF-IDF (unigrams + bigrams) → Multinomial Naive Bayes |
| **RNN** | Tokenized sequence (max length 200) → Embedding → LSTM(64) → Dense, trained with early stopping |

## Results

| Model | Accuracy | Weighted F1 | Negative F1 | Neutral F1 | Positive F1 |
|---|---|---|---|---|---|
| Naive Bayes (TF-IDF) | 86.3% | 0.88 | 0.79 | 0.51 | 0.93 |
| RNN (LSTM) | 87.9% | 0.89 | 0.79 | 0.52 | 0.94 |

Both models perform similarly, with the RNN holding a modest edge. Positive reviews (the majority class) are classified well by both (F1 ≥ 0.93); Neutral is the weak point for both (F1 ≈ 0.51–0.52), reflecting that 3-star reviews are often genuinely mixed in sentiment rather than a modeling failure.

## Repo / Artifacts

Both trained models are saved with the exact preprocessing artifacts needed to reproduce inference on new text:

```
naive_bayes_model.pkl        # trained Multinomial Naive Bayes (joblib)
tfidf_vectorizer.pkl         # fitted TF-IDF vectorizer (joblib)
rnn_sentiment_model.keras    # trained RNN, modern Keras format
tokenizer.pkl                # fitted Keras tokenizer (joblib)
config.pkl                   # label_map, max_len, max_words (joblib)
```

A working demonstration front end and Flask API route were built on top of these saved artifacts.

## Tech Stack

`numpy` · `pandas` · `matplotlib` · `seaborn` · `nltk` · `scikit-learn` · `tensorflow` / `keras` · `joblib` · `Flask`

## Conclusion & Key Insights

Class imbalance — not model architecture — was the central challenge. Both a simple classical model and a deep learning model needed the same balancing strategy before they could learn the minority sentiment classes at all. That both approaches converged on similar performance (86–88% accuracy) suggests this is close to the practical ceiling for this dataset and preprocessing; further gains are more likely to come from better class-imbalance handling (e.g. class weighting instead of oversampling) or richer features than from a larger model.

- The **Neutral** class remains inherently the hardest to classify in both models, consistent with 3-star reviews being genuinely ambiguous rather than a fixable model error.
- The pipeline is production-shaped (saved, reusable artifacts) but would need confidence-based fallback handling before being used in an automated decision-making context, given the Neutral class's lower precision.

## Getting Started

```bash
pip install -r requirements.txt
python app.py   # launches the Flask demo API
```

Load the saved artifacts (`*.pkl`, `*.keras`) at inference time to reproduce the exact preprocessing used in training — do not re-fit the vectorizer or tokenizer on new data.
