import io
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")

#input schema
class HouseFeatures(BaseModel):
    MedInc : float = Field(gt=0, description="Median income in block group")
    HouseAge : float =Field(gt=0, description="Avarage age of home")
    AveRooms : float =Field(gt=0, description="")
    AveBedrms : float =Field(gt=0)
    Population : float =Field(gt=0)
    AveOccup : float =Field(gt=0)
    Latitude : float =Field(ge=32, le=42)
    Longitude :float =Field(ge=-124, le=-114)

#home
@app.get("/")
def home():
    return{
        "message":"California house prediction api",
        "status":"running",
        "endpoint":"send POST request to /predict"

    }  

@app.get("/health")
def health():
    return{
        "status":"running",
        "model":"RandomForestRegressor",
        "avg_error":"$35000"
    }  


#prediction
@app.post("/predict")
def predict(house: HouseFeatures):

    try:
        input = pd.DataFrame([{
                    
                    "MedInc": house.MedInc,
                    "HouseAge": house.HouseAge,
                    "AveRooms": house.AveRooms,
                    "AveBedrms": house.AveBedrms,
                    "Population": house.Population,
                    "AveOccup": house.AveOccup,
                    "Latitude": house.Latitude,
                    "Longitude": house.Longitude
                }])
        prediction = model.predict(input)[0]
        price_usd = prediction * 100000
        
        return {
                    "predicted_price": f"${price_usd:,.0f}",
                   "predicted_price_float": f"{price_usd:.2f}",
                    "Fidence_range":f"${price_usd - 35000:,.0f} - ${price_usd + 35000:,.0f}"}
         
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error occurred while making prediction: {str(e)}")
        
@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    content = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read CSV file")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    required_columns = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude",
    ]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"These columns are missing in the file: {missing_columns}",
        )

    if df[required_columns].isnull().any().any():
        raise HTTPException(status_code=400, detail="File contains missing values")

    try:
        predictions = model.predict(df[required_columns])
        df["predicted_price_usd"] = (predictions * 100000).round(0)
        df["predicted_price_usd"] = df["predicted_price_usd"].apply(lambda x: f"${x:,.0f}")

        output = df.to_csv(index=False)
        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while making predictions: {str(e)}")