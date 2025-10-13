from fastapi import FastAPI
from deploy.domain.domain import BankMarketingRequest, BankMarketingResponse
from deploy.service.bank_marketing_service import BankMarketingService

long_term_predictor_app = FastAPI()

@long_term_predictor_app.post('/predict')
async def predict_acceptance(request: BankMarketingRequest)-> BankMarketingResponse:
    return BankMarketingService().predict(request=request)