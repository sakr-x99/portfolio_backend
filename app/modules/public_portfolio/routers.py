from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from . import models, schemas

router = APIRouter()

@router.get("/projects", response_model=List[schemas.ProjectResponse])
def get_projects(db: Session = Depends(get_db)):
    return db.query(models.Project).all()

@router.post("/projects", response_model=schemas.ProjectResponse)
def create_project(project: schemas.ProjectCreate, db: Session = Depends(get_db)):
    db_project = models.Project(**project.model_dump())
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

@router.put("/projects/{project_id}", response_model=schemas.ProjectResponse)
def update_project(project_id: int, project: schemas.ProjectCreate, db: Session = Depends(get_db)):
    db_project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    for key, value in project.model_dump().items():
        setattr(db_project, key, value)
    
    db.commit()
    db.refresh(db_project)
    return db_project

@router.delete("/projects/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    db_project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    db.delete(db_project)
    db.commit()
    return {"message": "Project deleted"}

# --- Skills ---
@router.get("/skills", response_model=List[schemas.SkillResponse])
def get_skills(db: Session = Depends(get_db)):
    return db.query(models.Skill).all()

@router.post("/skills", response_model=schemas.SkillResponse)
def create_skill(skill: schemas.SkillCreate, db: Session = Depends(get_db)):
    db_skill = models.Skill(**skill.model_dump())
    db.add(db_skill)
    db.commit()
    db.refresh(db_skill)
    return db_skill

@router.put("/skills/{skill_id}", response_model=schemas.SkillResponse)
def update_skill(skill_id: int, skill: schemas.SkillCreate, db: Session = Depends(get_db)):
    db_skill = db.query(models.Skill).filter(models.Skill.id == skill_id).first()
    if not db_skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    for key, value in skill.model_dump().items():
        setattr(db_skill, key, value)
    db.commit()
    db.refresh(db_skill)
    return db_skill

@router.delete("/skills/{skill_id}")
def delete_skill(skill_id: int, db: Session = Depends(get_db)):
    db_skill = db.query(models.Skill).filter(models.Skill.id == skill_id).first()
    if not db_skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    db.delete(db_skill)
    db.commit()
    return {"message": "Skill deleted"}

# --- Experience ---
@router.get("/experiences", response_model=List[schemas.ExperienceResponse])
def get_experiences(db: Session = Depends(get_db)):
    return db.query(models.Experience).all()

@router.post("/experiences", response_model=schemas.ExperienceResponse)
def create_experience(exp: schemas.ExperienceCreate, db: Session = Depends(get_db)):
    from datetime import datetime
    data = exp.model_dump()
    data['start_date'] = datetime.strptime(data['start_date'], '%Y-%m-%d').date()
    data['end_date'] = datetime.strptime(data['end_date'], '%Y-%m-%d').date() if data['end_date'] else None
    db_exp = models.Experience(**data)
    db.add(db_exp)
    db.commit()
    db.refresh(db_exp)
    return db_exp

@router.put("/experiences/{exp_id}", response_model=schemas.ExperienceResponse)
def update_experience(exp_id: int, exp: schemas.ExperienceCreate, db: Session = Depends(get_db)):
    from datetime import datetime
    db_exp = db.query(models.Experience).filter(models.Experience.id == exp_id).first()
    if not db_exp:
        raise HTTPException(status_code=404, detail="Experience not found")
    
    data = exp.model_dump()
    data['start_date'] = datetime.strptime(data['start_date'], '%Y-%m-%d').date()
    data['end_date'] = datetime.strptime(data['end_date'], '%Y-%m-%d').date() if data['end_date'] else None
    
    for key, value in data.items():
        setattr(db_exp, key, value)
    
    db.commit()
    db.refresh(db_exp)
    return db_exp

@router.delete("/experiences/{exp_id}")
def delete_experience(exp_id: int, db: Session = Depends(get_db)):
    db_exp = db.query(models.Experience).filter(models.Experience.id == exp_id).first()
    if not db_exp:
        raise HTTPException(status_code=404, detail="Experience not found")
    db.delete(db_exp)
    db.commit()
    return {"message": "Experience deleted"}

# --- Education ---
@router.get("/education", response_model=List[schemas.EducationResponse])
def get_education(db: Session = Depends(get_db)):
    return db.query(models.Education).all()

@router.post("/education", response_model=schemas.EducationResponse)
def create_education(edu: schemas.EducationCreate, db: Session = Depends(get_db)):
    from datetime import datetime
    data = edu.model_dump()
    data['start_date'] = datetime.strptime(data['start_date'], '%Y-%m-%d').date()
    data['end_date'] = datetime.strptime(data['end_date'], '%Y-%m-%d').date() if data['end_date'] else None
    db_edu = models.Education(**data)
    db.add(db_edu)
    db.commit()
    db.refresh(db_edu)
    return db_edu

@router.put("/education/{edu_id}", response_model=schemas.EducationResponse)
def update_education(edu_id: int, edu: schemas.EducationCreate, db: Session = Depends(get_db)):
    from datetime import datetime
    db_edu = db.query(models.Education).filter(models.Education.id == edu_id).first()
    if not db_edu:
        raise HTTPException(status_code=404, detail="Education not found")
    
    data = edu.model_dump()
    data['start_date'] = datetime.strptime(data['start_date'], '%Y-%m-%d').date()
    data['end_date'] = datetime.strptime(data['end_date'], '%Y-%m-%d').date() if data['end_date'] else None
    
    for key, value in data.items():
        setattr(db_edu, key, value)
    
    db.commit()
    db.refresh(db_edu)
    return db_edu

@router.delete("/education/{edu_id}")
def delete_education(edu_id: int, db: Session = Depends(get_db)):
    db_edu = db.query(models.Education).filter(models.Education.id == edu_id).first()
    if not db_edu:
        raise HTTPException(status_code=404, detail="Education not found")
    db.delete(db_edu)
    db.commit()
    return {"message": "Education deleted"}

# --- Services ---
@router.get("/services", response_model=List[schemas.ServiceResponse])
def get_services(db: Session = Depends(get_db)):
    return db.query(models.Service).all()

@router.post("/services", response_model=schemas.ServiceResponse)
def create_service(service: schemas.ServiceCreate, db: Session = Depends(get_db)):
    db_service = models.Service(**service.model_dump())
    db.add(db_service)
    db.commit()
    db.refresh(db_service)
    return db_service

@router.put("/services/{service_id}", response_model=schemas.ServiceResponse)
def update_service(service_id: int, service: schemas.ServiceCreate, db: Session = Depends(get_db)):
    db_service = db.query(models.Service).filter(models.Service.id == service_id).first()
    if not db_service:
        raise HTTPException(status_code=404, detail="Service not found")
    
    for key, value in service.model_dump().items():
        setattr(db_service, key, value)
    
    db.commit()
    db.refresh(db_service)
    return db_service

@router.delete("/services/{service_id}")
def delete_service(service_id: int, db: Session = Depends(get_db)):
    db_service = db.query(models.Service).filter(models.Service.id == service_id).first()
    if not db_service:
        raise HTTPException(status_code=404, detail="Service not found")
    db.delete(db_service)
    db.commit()
    return {"message": "Service deleted"}

# --- Articles ---
@router.get("/articles", response_model=List[schemas.ArticleResponse])
def get_articles(db: Session = Depends(get_db)):
    return db.query(models.Article).all()

@router.get("/articles/{slug}", response_model=schemas.ArticleResponse)
def get_article(slug: str, db: Session = Depends(get_db)):
    db_article = db.query(models.Article).filter(models.Article.slug == slug).first()
    if not db_article:
        raise HTTPException(status_code=404, detail="Article not found")
    return db_article

@router.post("/articles", response_model=schemas.ArticleResponse)
def create_article(article: schemas.ArticleCreate, db: Session = Depends(get_db)):
    from datetime import datetime
    data = article.model_dump()
    data['published_date'] = datetime.strptime(data['published_date'], '%Y-%m-%d').date()
    db_article = models.Article(**data)
    db.add(db_article)
    db.commit()
    db.refresh(db_article)
    return db_article

@router.delete("/articles/{article_id}")
def delete_article(article_id: int, db: Session = Depends(get_db)):
    db_article = db.query(models.Article).filter(models.Article.id == article_id).first()
    if not db_article:
        raise HTTPException(status_code=404, detail="Article not found")
    db.delete(db_article)
    db.commit()
    return {"message": "Article deleted"}
