from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    order_purchase_timestamp: str
    order_estimated_delivery_date: str

    item_count: float = Field(ge=0)
    product_count: float = Field(ge=0)
    seller_count: float = Field(ge=0)
    total_price: float = Field(ge=0)
    total_freight: float = Field(ge=0)
    payment_count: float = Field(ge=0)
    payment_value: float = Field(ge=0)
    max_installments: float = Field(ge=0)

    customer_state: str


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model: str
    version: str


class BatchPredictionRequest(BaseModel):
    orders: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]   