from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from . import models, schemas

router = APIRouter()

@router.get("/internal-projects", response_model=List[schemas.InternalProjectResponse])
def get_internal_projects(db: Session = Depends(get_db)):
    return db.query(models.InternalProject).all()

@router.post("/internal-projects", response_model=schemas.InternalProjectResponse)
def create_internal_project(project: schemas.InternalProjectCreate, db: Session = Depends(get_db)):
    db_project = models.InternalProject(**project.model_dump())
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

@router.get("/leads", response_model=List[schemas.LeadResponse])
def get_leads(db: Session = Depends(get_db)):
    return db.query(models.Lead).order_by(models.Lead.id.desc()).all()
