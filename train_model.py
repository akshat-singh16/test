# Replace with full train_model.py from canvas
"""train_model.py
Train TF-IDF + Logistic Regression and save model.pkl and vectorizer.pkl
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib




def preprocess_text(s: pd.Series) -> pd.Series:
return s.fillna('').astype(str).str.lower().str.replace(r'[^a-z\s]', ' ', regex=True)




def main(input_csv, model_out, vec_out, test_size=0.2, random_state=42):
df = pd.read_csv(input_csv)
if 'description' not in df.columns or 'category' not in df.columns:
raise ValueError('CSV must contain description and category columns')


X = preprocess_text(df['description'])
y = df['category']


# Remove categories with very few samples
vc = y.value_counts()
valid = vc[vc >= 2].index
df = df[df['category'].isin(valid)]
X = preprocess_text(df['description'])
y = df['category']


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)


tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=20000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


joblib.dump(model, model_out)
joblib.dump(tfidf, vec_out)
print('Saved model to', model_out, 'and vectorizer to', vec_out)




if __name__ == '__main__':
p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--model_out', default='model.pkl')
p.add_argument('--vec_out', default='vectorizer.pkl')
args = p.parse_args()
main(args.input, args.model_out, args.vec_out)
