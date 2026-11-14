import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# STEP 1: Load Dataset
df = pd.read_csv(
    r"C:\nlp\MARI-AI\data\reviews.csv"
)

print("Dataset Loaded Successfully!")


# STEP 2: Text Cleaning
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


df["cleaned_review"] = df["review"].apply(clean_text)


# STEP 3: Input and Output
X = df["cleaned_review"]

y = df["sentiment"]


# STEP 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# STEP 5: Create AI Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])


# STEP 6: Train Model
model.fit(X_train, y_train)

print("\nModel Training Completed!")


# STEP 7: Evaluate Model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))


# STEP 8: Save Model
joblib.dump(model, "sentiment_model.pkl")

print("\nModel Saved Successfully!")


# STEP 9: Test New Reviews
test_reviews = [
    "The food is amazing",
    "The room is very dirty",
    "The service is average"
]

print("\nAI Predictions:")

for review in test_reviews:

    cleaned = clean_text(review)

    result = model.predict([cleaned])

    print(review, "->", result[0])