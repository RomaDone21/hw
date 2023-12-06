from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import tempfile
import pandas as pd
import joblib

model = joblib.load('ridge_model.pkl')

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    
    prediction = [model.predict(obj) for obj in item.objects]
    
    return prediction


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    
    predictions = [model.predict(obj) for obj in items.objects]
    
    return predictions


@app.post("/predict_items_csv")
def predict_items_csv(items: Items) -> FileResponse:
    
    predictions = [model.predict(obj) for obj in items.objects]

    with NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        df = pd.DataFrame({"Prediction": predictions})  
        df.to_csv(temp_file.name, index=False)


    return FileResponse(temp_file.name, filename="predictions.csv")