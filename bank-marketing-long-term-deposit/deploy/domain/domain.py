from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class BankMarketingRequest(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str

class BankMarketingResponse(BaseModel):
    y: int