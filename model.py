import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import pickle

# Load the data
tourist_df = pd.read_csv('all_places.csv')

# Define the parameter grid for GridSearchCV
param_grid_rf = {
    'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_depth': [None, 10, 20],
}

# Define the model pipeline for RandomForestClassifier
model_rf = make_pipeline(TfidfVectorizer(), RandomForestClassifier())

# Perform grid search
grid_search_rf = GridSearchCV(model_rf, param_grid_rf, cv=5)
grid_search_rf.fit(tourist_df['description'], tourist_df['city'])

# Save the trained model using pickle
with open("classifier.pkl", "wb") as f:
    pickle.dump(grid_search_rf, f)

print("Model training and saving completed.")
