# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
import re
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.utils import resample

nltk.download('punkt_tab')
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load the dataset
df = pd.read_csv("Reviews.csv")
print(df.head(10))

# ### Step 3: Data Cleaning

# Explore the dataset
df.info()

df.describe()

# Handle missing values
df.isnull().sum()

df[df["ProfileName"].isnull()]

cleaned_df = df.dropna()

cleaned_df.isnull().sum()

# Remove duplicates
cleaned_df.duplicated().sum()

# Correct data types
cleaned_df["Time"] = pd.to_datetime(cleaned_df["Time"], unit="s")

print(cleaned_df["Time"].head())
print(cleaned_df["Time"].dtype)

# To check whether the HelpfulnessNumerator exceeds the HelpfulnessDenominator which is logically impossible
invalid_helpfulness = cleaned_df[cleaned_df["HelpfulnessNumerator"] > cleaned_df["HelpfulnessDenominator"]]
print(invalid_helpfulness)

cleaned_df[cleaned_df["HelpfulnessNumerator"] > cleaned_df["HelpfulnessDenominator"]].shape[0]

cleaned_df = cleaned_df[cleaned_df["HelpfulnessNumerator"] <= cleaned_df["HelpfulnessDenominator"]]
cleaned_df[cleaned_df["HelpfulnessNumerator"] > cleaned_df["HelpfulnessDenominator"]].shape[0]

# Sorting data according to ProductId in ascending order
sorted_df = cleaned_df.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort')
sorted_df.shape

sorted_df["Score"].unique()

# Give reviews with Score>3 a positive rating, reviews with a score<3 a negative rating, and reviews with a score=3 a neutral rating
def label_sentiment(score):
    if score > 3:
        return "positive"
    elif score < 3:
        return "negative"
    else:
        return "neutral"

sorted_df["sentiment"] = sorted_df["Score"].apply(label_sentiment)

sorted_df.head(10)

# Checking to see how much % of data still remains
(sorted_df['Id'].size*1.0)/(df['Id'].size*1.0)*100


# Box plots of scores vs HelpfulnessNumerator & HelpfulnessDenominator (log scale)
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
sns.boxplot(x="Score", y="HelpfulnessNumerator", data=sorted_df)
plt.yscale('log')
plt.title("Box Plot of Scores vs HelpfulnessNumerator (log scale)")
plt.xlabel("Score")
plt.ylabel("HelpfulnessNumerator Score")

plt.subplot(1, 2, 2)
sns.boxplot(x="Score", y="HelpfulnessDenominator", data=sorted_df)
plt.yscale('log')
plt.title("Box Plot of Scores vs HelpfulnessDenominator (log scale)")
plt.xlabel("Score")
plt.ylabel("HelpfulnessDenominator Score")

plt.tight_layout()
plt.show()

# Histogram of the Scores
sns.histplot(data=sorted_df, x="Score", kde=True, bins=range(int(sorted_df["Score"].min()), int(sorted_df["Score"].max()) + 2, 1))
plt.title("Distribution of Scores")
plt.ticklabel_format(style='plain', axis='y')
plt.show()

# Scatter plot to see the relationship between HelpfulnessNumerator & HelpfulnessDenominator
plt.scatter(sorted_df["HelpfulnessNumerator"], sorted_df["HelpfulnessDenominator"])
plt.title("Scatter Plot of HelpfulnessNumerator vs HelpfulnessDenominator")
plt.xlabel("HelpfulnessNumerator Score")
plt.ylabel("HelpfulnessDenominator Score")

# Check the number of Division-by-zero entries
print((sorted_df["HelpfulnessDenominator"] == 0).sum()) 

# Explicitly filter NaN (Division-by-zero)
reviews_with_votes = sorted_df[sorted_df["HelpfulnessDenominator"] > 0].copy()

reviews_with_votes["helpfulness_score"] = (reviews_with_votes["HelpfulnessNumerator"] / reviews_with_votes["HelpfulnessDenominator"])

# Line plot of Scores vs Helpfulness Score
helpfulness_score_grouped = reviews_with_votes.groupby("Score")["helpfulness_score"].mean().sort_values()

plt.plot(helpfulness_score_grouped.index, helpfulness_score_grouped.values)
plt.title("Line plot of Score vs Helpfulness Score (reviews with ≥1 vote only)")
plt.xlabel("Score")
plt.ylabel("Helpfulness Score")
plt.show()

# Text Preprocessing
# set of stopwords
stop = set(stopwords.words('english'))

# Initialization of Lemmatizer
lemmatizer = WordNetLemmatizer()

# function to preprocess the text + summary features
def preprocess(text, stopwords_set):
    text = re.sub('<.*?>', ' ', text)                      # remove HTML tags
    text = re.sub(r'[?!\'"#]', '', text)                    # remove punctuation
    text = re.sub(r'[.,)(|/]', ' ', text)                    # replace separators with space
    words = text.split()
    cleaned = [
        lemmatizer.lemmatize(w.lower())
        for w in words 
        if w.isalpha() and len(w) > 2 and w.lower() not in stopwords_set
    ]
    return " ".join(cleaned)

# Applying the text preprocessing to the most important features (Text & Summary)
sorted_df['clean_text'] = sorted_df['Text'].apply(lambda x: preprocess(x, stop))
sorted_df['clean_summary'] = sorted_df['Summary'].apply(lambda x: preprocess(x, stop))

sorted_df.head(10)

sorted_df['clean_combined'] = sorted_df['clean_summary'] + " " + sorted_df['clean_text']

X = sorted_df['clean_combined']
y = sorted_df['sentiment']

# Split before upsampling — to avoid data leakage
X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

train_df = pd.DataFrame({'clean_combined': X_train_text, 'sentiment': y_train})

# upsampling to balance classes (train set only)
df_neg = train_df[train_df['sentiment']=='negative']
df_neu = train_df[train_df['sentiment']=='neutral']
df_pos = train_df[train_df['sentiment']=='positive']

max_size = max(len(df_neg), len(df_neu), len(df_pos))

df_neg_up = resample(df_neg, replace=True, n_samples=max_size, random_state=40)
df_neu_up = resample(df_neu, replace=True, n_samples=max_size, random_state=40)
df_pos_up = resample(df_pos, replace=True, n_samples=max_size, random_state=40)

train_balanced = pd.concat([df_neg_up, df_neu_up, df_pos_up]).sample(frac=1, random_state=40).reset_index(drop=True)

print(train_balanced['sentiment'].value_counts())

X_train_text_balanced = train_balanced['clean_combined']
y_train_balanced = train_balanced['sentiment']

# Tokenization & Pads sequences (for RNN)
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train_text_balanced)   # fit only on balanced Training text

X_train_seq = tokenizer.texts_to_sequences(X_train_text_balanced)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)   # test text 

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# TF-IDF
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
X_train = tf_idf_vect.fit_transform(X_train_text_balanced)
X_test = tf_idf_vect.transform(X_test_text)   # transform only

features = tf_idf_vect.get_feature_names_out()
len(features)

features[90000:90010]

label_map = {"negative": 0, "neutral": 1, "positive": 2}

y_train_encoded = y_train_balanced.map(label_map)
y_test_encoded = y_test.map(label_map)

# One-hot encoding (for RNN)
y_train = to_categorical(y_train_encoded, num_classes=3)
y_test = to_categorical(y_test_encoded, num_classes=3)

# train
model = MultinomialNB()
model.fit(X_train, y_train_encoded)

# predict (Naive Bayes)
y_pred = model.predict(X_test)

# RNN
rnn_model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')   # 3 classes: negative, neutral, positive
])

# Compile LSTM model
rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train RNN model
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',      # watch validation loss
    patience=2,              # stop if it doesn't improve for 2 epochs in a row
    restore_best_weights=True  # roll back to the best epoch's weights, not the last one
)

history = rnn_model.fit(
    X_train_pad, y_train,
    validation_data=(X_test_pad, y_test),
    epochs=5,               # set a higher ceiling — early stopping will cut it short if needed
    batch_size=256,
    callbacks=[early_stop]
)

# Evaluate Naive Bayes model
cm = confusion_matrix(y_test_encoded, y_pred)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test_encoded, y_pred, target_names=["negative","neutral","positive"]))
print("Test Accuracy:", accuracy_score(y_test_encoded, y_pred))

# plotting the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["negative","neutral","positive"],
            yticklabels=["negative","neutral","positive"])
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.title("Confusion Matrix")
plt.show()

# Evaluate RNN model
y_pred_rnn_probs = rnn_model.predict(X_test_pad)
y_pred_rnn = np.argmax(y_pred_rnn_probs, axis=1)
y_test_rnn_labels = np.argmax(y_test, axis=1)   # convert one-hot back to plain labels for comparison

cm_rnn = confusion_matrix(y_test_rnn_labels, y_pred_rnn)
print("RNN Confusion Matrix:\n", cm_rnn)
print("RNN Classification Report:\n", classification_report(y_test_rnn_labels, y_pred_rnn, target_names=["negative","neutral","positive"]))
print("RNN Test Accuracy:", accuracy_score(y_test_rnn_labels, y_pred_rnn))

plt.figure(figsize=(6,5))
sns.heatmap(cm_rnn, annot=True, fmt='d', cmap='Greens',
            xticklabels=["negative","neutral","positive"],
            yticklabels=["negative","neutral","positive"])
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.title("RNN Confusion Matrix")
plt.show()

# New Predictions

# Preprocessing step
def prepare_review(text, summary=""):
    clean_text = preprocess(text, stop)
    clean_summary = preprocess(summary, stop)
    return clean_summary + " " + clean_text
predicted = prepare_review("This product was amazing, I loved it!")

# Predict using Naive Bayes (TF-IDF)
def predict_sentiment_nb(text, summary=""):
    cleaned = prepare_review(text, summary)
    vector = tf_idf_vect.transform([cleaned])   # transform only — reuse the FITTED vectorizer
    pred = model.predict(vector)[0]

    label_map_inv = {0: "negative", 1: "neutral", 2: "positive"}
    return label_map_inv[pred]

print(predict_sentiment_nb(predicted))
print(predict_sentiment_nb("Terrible quality, broke after one use."))

# Predict using the RNN
def predict_sentiment_rnn(text, summary=""):
    cleaned = prepare_review(text, summary)
    seq = tokenizer.texts_to_sequences([cleaned])   # reuse the FITTED tokenizer, no fit_on_texts here
    padded = pad_sequences(seq, maxlen=max_len)

    pred_probs = rnn_model.predict(padded, verbose=0)
    pred_class = np.argmax(pred_probs, axis=1)[0]

    label_map_inv = {0: "negative", 1: "neutral", 2: "positive"}
    return label_map_inv[pred_class]

print(predict_sentiment_rnn("This product was amazing, I loved it!"))
print(predict_sentiment_rnn("Terrible quality, broke after one use."))

# Productionization and deployment of models

import joblib

# Save the Naive Bayes model + TF-IDF vectorizer (joblib)
joblib.dump(model, "naive_bayes_model.pkl")
joblib.dump(tf_idf_vect, "tfidf_vectorizer.pkl")

# Save the RNN model (use Keras's own .save(), not joblib)
rnn_model.save("rnn_sentiment_model.keras")   # modern Keras format

# Save the Tokenizer (needed for RNN preprocessing)
joblib.dump(tokenizer, "tokenizer.pkl")

# Save label_map and max_len
joblib.dump({"label_map": label_map, "max_len": max_len, "max_words": max_words}, "config.pkl")

