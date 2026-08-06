from flask import Flask, jsonify, request, render_template
import os
import json
import re
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)

MAX_REVIEW_CHARS = 5000  # basic guard against huge/abusive payloads



# Load every saved artifact ONCE at startup, not per-request.
# (The tutorial snippet you shared calls joblib.load() inside predict() on
# every single request — that re-reads the pickle from disk each time,
# which is slow and pointless since the model never changes between calls.)

# Vercel's serverless filesystem is read-only except /tmp, so nltk.download()
# to the default location crashes the function on every cold start. Point it
# at /tmp instead, and skip the download if it's already there.
NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

for resource, lookup_path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
    try:
        nltk.data.find(lookup_path)
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_DIR, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

nb_model = joblib.load("naive_bayes_model.pkl")
tf_idf_vect = joblib.load("tfidf_vectorizer.pkl")

rnn_interpreter = Interpreter(model_path="rnn_sentiment_model.tflite")
rnn_interpreter.allocate_tensors()
RNN_INPUT_DETAILS = rnn_interpreter.get_input_details()[0]
RNN_OUTPUT_DETAILS = rnn_interpreter.get_output_details()[0]

# tokenizer_vocab.json is a plain extract of the original Keras Tokenizer's
# word_index/config — avoids needing the `keras` package just to unpickle it.
with open("tokenizer_vocab.json") as f:
    _vocab = json.load(f)
WORD_INDEX = _vocab["word_index"]
NUM_WORDS = _vocab["num_words"]
TOK_FILTERS = _vocab["filters"]
TOK_LOWER = _vocab["lower"]
_TRANSLATE_MAP = str.maketrans({c: " " for c in TOK_FILTERS})


def texts_to_sequence(text):
    """Reimplementation of Keras Tokenizer.texts_to_sequences for one string."""
    if TOK_LOWER:
        text = text.lower()
    words = [w for w in text.translate(_TRANSLATE_MAP).split(" ") if w]
    return [
        idx for w in words
        if (idx := WORD_INDEX.get(w)) is not None and (NUM_WORDS is None or idx < NUM_WORDS)
    ]


def pad_sequence(seq, maxlen):
    """Reimplementation of Keras pad_sequences (padding='pre', truncating='pre')."""
    seq = seq[-maxlen:]
    padded = [0] * (maxlen - len(seq)) + seq
    return np.array([padded], dtype=np.float32)

config = joblib.load("config.pkl")
LABEL_MAP = config["label_map"]                       # {"negative": 0, "neutral": 1, "positive": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}   # {0: "negative", 1: "neutral", 2: "positive"}
MAX_LEN = config["max_len"]



# Same preprocessing used to train both models — must stay identical to
# model.py's preprocess()/prepare_review(), or train/inference will skew.

def preprocess(text, stopwords_set):
    text = re.sub("<.*?>", " ", text)             # remove HTML tags
    text = re.sub(r"[?!\'\"#]", "", text)          # remove punctuation
    text = re.sub(r"[.,)(|/]", " ", text)          # replace separators with space
    words = text.split()
    cleaned = [
        LEMMATIZER.lemmatize(w.lower())
        for w in words
        if w.isalpha() and len(w) > 2 and w.lower() not in stopwords_set
    ]
    return " ".join(cleaned)


def prepare_review(text, summary=""):
    clean_text = preprocess(text, STOP_WORDS)
    clean_summary = preprocess(summary, STOP_WORDS)
    return f"{clean_summary} {clean_text}".strip()


def get_review_text():
    """Pull review_text from form or JSON body, with basic validation."""
    if request.is_json:
        text = (request.get_json(silent=True) or {}).get("review_text", "")
    else:
        text = request.form.get("review_text", "")
    text = (text or "").strip()
    if not text:
        return None, ("Review text is empty.", 400)
    if len(text) > MAX_REVIEW_CHARS:
        return None, (f"Review text is too long (max {MAX_REVIEW_CHARS} characters).", 400)
    return text, None



# Routes

@app.route("/")
def hello_world():
    return 'Hello World!'

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/predict-nb", methods=["POST"])
def predict_nb():
    text, error = get_review_text()
    if error:
        message, status = error
        return jsonify({"error": message}), status

    try:
        cleaned = prepare_review(text)
        vector = tf_idf_vect.transform([cleaned])
        pred = nb_model.predict(vector)[0]
        probs = nb_model.predict_proba(vector)[0]  # [P(negative), P(neutral), P(positive)]

        return jsonify({
            "label": LABEL_MAP_INV[pred],
            "pos": float(probs[LABEL_MAP["positive"]]),
            "neu": float(probs[LABEL_MAP["neutral"]]),
            "neg": float(probs[LABEL_MAP["negative"]]),
        })
    except Exception:
        app.logger.exception("NB prediction failed")
        return jsonify({"error": "Could not score this review right now."}), 500


@app.route("/predict-rnn", methods=["POST"])
def predict_rnn():
    text, error = get_review_text()
    if error:
        message, status = error
        return jsonify({"error": message}), status

    try:
        cleaned = prepare_review(text)
        seq = texts_to_sequence(cleaned)
        padded = pad_sequence(seq, maxlen=MAX_LEN)

        rnn_interpreter.set_tensor(RNN_INPUT_DETAILS["index"], padded)
        rnn_interpreter.invoke()
        probs = rnn_interpreter.get_tensor(RNN_OUTPUT_DETAILS["index"])[0]  # [P(negative), P(neutral), P(positive)]
        pred_class = int(np.argmax(probs))

        return jsonify({
            "label": LABEL_MAP_INV[pred_class],
            "pos": float(probs[LABEL_MAP["positive"]]),
            "neu": float(probs[LABEL_MAP["neutral"]]),
            "neg": float(probs[LABEL_MAP["negative"]]),
        })
    except Exception:
        app.logger.exception("RNN prediction failed")
        return jsonify({"error": "Could not score this review right now."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
