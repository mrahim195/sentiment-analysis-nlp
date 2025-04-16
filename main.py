import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('english')

# Load data
df = pd.read_csv('Tweets.csv')  # Make sure file is in the same folder
print(df.head())

# Use only text and sentiment columns
df = df[['text', 'airline_sentiment']]
df.columns = ['text', 'label']

# Text preprocessing
df['text'] = df['text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# Vectorization (convert text to numbers)
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
