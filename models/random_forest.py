import sys
import os

sys.path.append(os.path.abspath(os.path.join('..', '/Users/benedictzuzi/Documents/nlp_sentiment_analysis/src')))
from preprocess import preprocess


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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

# Vectorize the text data
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))  # Using both unigrams and bigrams
X_train_vect = tfidf_vectorizer.fit_transform(X_train)
X_test_vect = tfidf_vectorizer.transform(X_test)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# Initialize GridSearchCV with the Random Forest model and the hyperparameter grid
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')

# Fit the GridSearchCV object to the vectorized training data
grid_search.fit(X_train_vect, y_train)

# Retrieve the best estimator found by the grid search
best_rf_model = grid_search.best_estimator_

# Print the best parameters found by the grid search
print(f"Best parameters found: {grid_search.best_params_}")


# Train the Random Forest Classifier
#rf_model.fit(X_train_vect, y_train)

# Predict on the training data
#train_predictions = rf_model.predict(X_train_vect)

# Predict on the test data
test_predictions = best_rf_model.predict(X_test_vect)

# Evaluate the model's performance
test_accuracy = accuracy_score(y_test, test_predictions)

# Print out performance metrics
print(f"Test Accuracy: {test_accuracy}")
print(classification_report(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))

