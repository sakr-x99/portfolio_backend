from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.core.auth import get_current_user
from . import models, schemas

router = APIRouter(dependencies=[Depends(get_current_user)])

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

@router.put("/internal-projects/{project_id}", response_model=schemas.InternalProjectResponse)
def update_internal_project(project_id: int, project: schemas.InternalProjectCreate, db: Session = Depends(get_db)):
    db_project = db.query(models.InternalProject).filter(models.InternalProject.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    for key, value in project.model_dump().items():
        setattr(db_project, key, value)
    
    db.commit()
    db.refresh(db_project)
    return db_project

@router.delete("/internal-projects/{project_id}")
def delete_internal_project(project_id: int, db: Session = Depends(get_db)):
    db_project = db.query(models.InternalProject).filter(models.InternalProject.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db.delete(db_project)
    db.commit()
    return {"message": "Project deleted successfully"}

@router.get("/leads", response_model=List[schemas.LeadResponse])
def get_leads(db: Session = Depends(get_db)):
    return db.query(models.Lead).order_by(models.Lead.id.desc()).all()
