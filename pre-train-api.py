from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from fastapi.exceptions import HTTPException

app = FastAPI()

class ScoringItem(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load the Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Create and train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_iris, y_iris)

@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    try:
        input_data = [[item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]]
        df = pd.DataFrame(input_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        prediction = model.predict(df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
