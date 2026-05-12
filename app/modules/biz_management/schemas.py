from pydantic import BaseModel
from typing import Optional

class InternalProjectBase(BaseModel):
    name: str
    business_model: str
    features: str
    tech_stack: str
    initial_cost: float
    operational_cost_per_month: float
    marketing_plan: str
    status: str

class InternalProjectCreate(InternalProjectBase):
    pass

class InternalProjectResponse(InternalProjectBase):
    id: int

    class Config:
        from_attributes = True

class LeadBase(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    inquiry_type: str
    message: Optional[str] = None
    meeting_time: Optional[str] = None
    summary: Optional[str] = None
    created_at: Optional[str] = None

class LeadResponse(LeadBase):
    id: int

    class Config:
        from_attributes = True
