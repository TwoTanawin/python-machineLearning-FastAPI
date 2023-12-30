# Bring in lightweight dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from fastapi import HTTPException


app = FastAPI()

class ScoringItem(BaseModel): 
    YearsAtCompany: float #/ 1, // Float value 
    EmployeeSatisfaction: float #0.01, // Float value 
    Position:str # "Non-Manager", # Manager or Non-Manager
    Salary: int #4.0 // Ordinal 1,2,3,4,5

# Assuming model is an instance of RandomForestClassifier
model = RandomForestClassifier()

with open('/app/fast-api/fast-ml/model/rfmodel.pkl', 'rb') as f: 
    model = pickle.load(f)

# @app.post('/')
# async def scoring_endpoint(item:ScoringItem): 
#     df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
#     yhat = model.predict(df)
#     return {"prediction":int(yhat)}

@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    try:
        df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
        yhat = model.predict(df)
        return {"prediction": int(yhat)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))