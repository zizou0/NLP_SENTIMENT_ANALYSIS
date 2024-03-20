import sys
import os

sys.path.append(os.path.abspath(os.path.join('..', '/Users/benedictzuzi/Documents/nlp_sentiment_analysis/src')))
from preprocess import preprocess


from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from joblib import dump, load

filename = '/Users/benedictzuzi/Documents/nlp_sentiment_analysis/data/imdb_dataset.csv'

df = preprocess(filename)  
sentiments_encoded = df['sentiment'].map({'positive': 1, 'negative': 0})

X = df['cleaned_review']
y = sentiments_encoded


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))  # Using both unigrams and bigrams
X_train_vect = tfidf_vectorizer.fit_transform(X_train)
X_test_vect = tfidf_vectorizer.transform(X_test)

# Initialize the XGBoost Classifier
xgb_model = XGBClassifier(eval_metric='logloss')

# Define a parameter grid for GridSearch
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, .05, 0.01],
    'max_depth': [3, 5],
    'subsample': [0.7, 1],
    'colsample_bytree': [0.7, 1],
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, verbose=1, scoring='accuracy')

# Fit the GridSearchCV object to find the best parameters
grid_search.fit(X_train_vect, y_train)

# Retrieve the best estimator
best_xgb_model = grid_search.best_estimator_

# Make predictions using the test set
test_predictions = best_xgb_model.predict(X_test_vect)

# Calculate accuracy on the test set
test_accuracy = accuracy_score(y_test, test_predictions)

# Print out performance metrics
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Test accuracy: {test_accuracy}")
print(classification_report(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))

