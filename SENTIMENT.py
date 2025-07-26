import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = {
    'review': [
        "I absolutely love this product!", "It stopped working after two days.",
        "The features are amazing and easy to use.", "Worst purchase I've made this year.",
        "Everything works perfectly, thank you!", "Completely useless and frustrating.",
        "Excellent design and top-notch quality.", "The material feels so cheap.",
        "Fast delivery and great service!", "I can't believe how badly it's made.",
        "Fits perfectly and works like a charm.", "Broken when it arrived.",
        "Highly recommend to anyone!", "Very disappointing and overhyped.",
        "Incredible build quality and finish.", "Do not waste your money on this.",
        "Reliable and efficient, just what I needed.", "Not what I expected at all.",
        "Really happy with the performance.", "One of the worst things I've ordered.",
        "Super helpful customer support.", "It doesn't function correctly.",
        "Would buy again in a heartbeat.", "Looks nice but doesn’t perform well.",
        "Absolutely worth the price!", "Save your money and skip this.",
        "Totally satisfied with the results.", "It broke after first use.",
        "Exceptional experience overall!", "Terrible battery life.",
        "Beautifully crafted and sturdy.", "Too many issues to deal with.",
        "Great for daily use and travel.", "Not durable whatsoever.",
        "Smooth performance every time.", "The instructions were confusing.",
        "A flawless purchase!", "It just doesn’t work properly.",
        "Amazing deal for the features!", "Not satisfied with the quality."
    ],
    'sentiment': [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
                  1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
}

df = pd.DataFrame(data)


X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Negative", "Positive"], zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))