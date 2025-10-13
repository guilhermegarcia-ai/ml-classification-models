import pandas as pd
from deploy.domain.domain import BankMarketingRequest, BankMarketingResponse
from deploy.utils import utils

class BankMarketingService():
    def __init__(self):
        self.model_pipeline_path = 'deploy/artifacts/xgb_pipeline.pkl'
        self.final_pipeline = utils.load_artifact(self.model_pipeline_path)
    

    def predict(self, request: BankMarketingRequest) -> BankMarketingResponse:
        """
        """
        df = pd.DataFrame([request.model_dump()])
        
        prediction = self.final_pipeline.predict(df)[0]
        
        response = BankMarketingResponse
        response.y = prediction

        return response
    
# ------------------------
# Testing (python -m deploy.service.bank_marketing_service)
# if __name__ == "__main__":
#     test_request = BankMarketingRequest(age = 38,
#                                   job = 'entrepreneur',
#                                   marital = 'single',
#                                   education = 'tertiary',
#                                   default = 'no',
#                                   balance = 243,
#                                   housing = 'no',
#                                   loan = 'yes',
#                                   contact = 'unknown',
#                                   day = 5,
#                                   month = 'may',
#                                   campaign = 1,
#                                   pdays = -1,
#                                   previous = 0,
#                                   poutcome = 'unknown'
#     )

#     new_used_service = BankMarketingService()
#     response = new_used_service.predict(request= test_request)
#     print(response.y)