from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime

class ComplaintSource(BaseModel):
    tags: Optional[str] = None
    zip_code: Optional[str]
    complaint_id: str
    issue: str
    date_received: datetime
    state: str = Field(..., max_length=2)  # State abbreviation validation
    consumer_disputed: Optional[str] = None
    product: str
    company_response: str
    company: str
    submitted_via: str
    date_sent_to_company: datetime
    company_public_response: Optional[str] = None
    sub_product: Optional[str] = None
    timely: str
    complaint_what_happened: Optional[str] = None
    sub_issue: Optional[str] = None
    consumer_consent_provided: Optional[str] = None

    @field_validator("timely")
    def validate_timely(cls, v):
        if v not in ["Yes", "No"]:
            raise ValueError("timely must be 'Yes' or 'No'")
        return v

class ComplaintData(BaseModel):
    index: str
    type: str
    id: str
    score: float
    source: ComplaintSource

class TopicPrediction(BaseModel):
    complaint_id: int
    complaint_text: str
    prediction: float

class PredictionResponse(BaseModel):
    data: Optional[List[TopicPrediction]] = None
    error: Optional[str] = None