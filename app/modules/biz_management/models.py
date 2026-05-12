from sqlalchemy import Column, Integer, String, Text, Float, Date
from app.core.database import Base

class InternalProject(Base):
    __tablename__ = "biz_internal_projects"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    business_model = Column(Text)
    features = Column(Text)
    tech_stack = Column(String)
    
    # Financials
    initial_cost = Column(Float, default=0.0)
    operational_cost_per_month = Column(Float, default=0.0)
    
    # Marketing
    marketing_plan = Column(Text)
    
    # Status
    status = Column(String, default="Planning") # Planning, Development, Active, Archived
    start_date = Column(Date, nullable=True)

class Lead(Base):
    __tablename__ = "biz_leads"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=True)
    email = Column(String, index=True, nullable=True)
    phone = Column(String, nullable=True)
    inquiry_type = Column(String) # Hiring, Freelance, General
    message = Column(Text, nullable=True)
    meeting_time = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    created_at = Column(String, nullable=True) # ISO format
