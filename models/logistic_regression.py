import sys
import os

sys.path.append(os.path.abspath(os.path.join('..', '/Users/benedictzuzi/Documents/nlp_sentiment_analysis/src')))
from preprocess import preprocess


import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from joblib import dump, load

filename = '/Users/benedictzuzi/Documents/nlp_sentiment_analysis/data/imdb_dataset.csv'

df = preprocess(filename)  
sentiments_encoded = df['sentiment'].map({'positive': 1, 'negative': 0})

X = df['cleaned_review']
y = sentiments_encoded

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32, stratify=y)

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_train_vect = tfidf_vectorizer.fit_transform(X_train)
X_test_vect = tfidf_vectorizer.transform(X_test)

# Initialize the Logistic Regression model
model = LogisticRegression(penalty=None, max_iter=1000)



#v K-Fold Cross-Validation (comment out to utilize)
# pipeline = make_pipeline(tfidf_vectorizer, model)
# scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')  # cv is the number of folds
# print(f"Cross-validation scores for each fold: {scores}")
# # Print the mean accuracy and the 95% confidence interval of the score estimate
# print(f"Mean cross-validation score (k=5): {scores.mean()}")
# print(f"95% confidence interval for cross-validation score: {scores.mean()} +/- {scores.std() * 2}")




#print(X_train_vect.shape)

# Train the model
model.fit(X_train_vect, y_train)


# Make predictions using the test set
train_predictions = model.predict(X_train_vect)
test_predictions = model.predict(X_test_vect)

print(f"Train accuracy: {accuracy_score(y_train, train_predictions)}")


# Evaluate the model's performance
accuracy = accuracy_score(y_test, test_predictions)
print(f'Class Balance: {sentiments_encoded.value_counts()}')
print(f'Test accuracy: {accuracy}')
print(classification_report(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))