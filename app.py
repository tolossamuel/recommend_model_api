import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle


# Define a request body model
class TextInput(BaseModel):
    query: str

# Load the trained model
with open("classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

# Create FastAPI app
app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/predict')
def predict_city(input: TextInput):
    query = input.query
    predicted_probs = classifier.predict_proba([query])[0]
    classes = classifier.classes_
    city_prob_pairs = list(zip(classes, predicted_probs))
    sorted_city_prob_pairs = sorted(city_prob_pairs, key=lambda x: x[1], reverse=True)
    sorted_cities = [pair[0] for pair in sorted_city_prob_pairs if pair[1] >= 0.0]
    return sorted_cities[:10]

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
