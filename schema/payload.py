from pydantic import BaseModel

class GenerateRepuest(BaseModel):
    month : int
    temp : float
    dew : float
    humidity : float
    sealevelpressure : float
    winddir : float
    solarradiation : float
    windspeed : float
    precipprob : int

class GenerateResponse(BaseModel):
    pred_preciptype : str
    prob_preciptype : float