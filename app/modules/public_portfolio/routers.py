from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.core.cache import cached
from app.core.crud import CRUDBase
from app.core.auth import get_current_user
from . import models, schemas

router = APIRouter()

project_crud = CRUDBase(models.Project, "get_projects")
skill_crud = CRUDBase(models.Skill, "get_skills")
experience_crud = CRUDBase(models.Experience, "get_experiences")
education_crud = CRUDBase(models.Education, "get_education")
service_crud = CRUDBase(models.Service, "get_services")
article_crud = CRUDBase(models.Article, "get_articles")

# ── Projects ──────────────────────────────────────────────────────────────────
@router.get("/projects", response_model=List[schemas.ProjectResponse])
@cached(ttl=120)
def get_projects(db: Session = Depends(get_db)):
    return project_crud.get_all(db)

@router.post("/projects", response_model=schemas.ProjectResponse, dependencies=[Depends(get_current_user)])
def create_project(project: schemas.ProjectCreate, db: Session = Depends(get_db)):
    return project_crud.create(db, project)

@router.put("/projects/{project_id}", response_model=schemas.ProjectResponse, dependencies=[Depends(get_current_user)])
def update_project(project_id: int, project: schemas.ProjectCreate, db: Session = Depends(get_db)):
    return project_crud.update(db, project_id, project)

@router.delete("/projects/{project_id}", dependencies=[Depends(get_current_user)])
def delete_project(project_id: int, db: Session = Depends(get_db)):
    return project_crud.delete(db, project_id)

# ── Skills ────────────────────────────────────────────────────────────────────
@router.get("/skills", response_model=List[schemas.SkillResponse])
@cached(ttl=300)
def get_skills(db: Session = Depends(get_db)):
    return skill_crud.get_all(db)

@router.post("/skills", response_model=schemas.SkillResponse, dependencies=[Depends(get_current_user)])
def create_skill(skill: schemas.SkillCreate, db: Session = Depends(get_db)):
    return skill_crud.create(db, skill)

@router.put("/skills/{skill_id}", response_model=schemas.SkillResponse, dependencies=[Depends(get_current_user)])
def update_skill(skill_id: int, skill: schemas.SkillCreate, db: Session = Depends(get_db)):
    return skill_crud.update(db, skill_id, skill)

@router.delete("/skills/{skill_id}", dependencies=[Depends(get_current_user)])
def delete_skill(skill_id: int, db: Session = Depends(get_db)):
    return skill_crud.delete(db, skill_id)

# ── Experience ────────────────────────────────────────────────────────────────
@router.get("/experiences", response_model=List[schemas.ExperienceResponse])
@cached(ttl=300)
def get_experiences(db: Session = Depends(get_db)):
    return experience_crud.get_all(db)

@router.post("/experiences", response_model=schemas.ExperienceResponse, dependencies=[Depends(get_current_user)])
def create_experience(exp: schemas.ExperienceCreate, db: Session = Depends(get_db)):
    return experience_crud.create(db, exp, date_fields=["start_date", "end_date"])

@router.put("/experiences/{exp_id}", response_model=schemas.ExperienceResponse, dependencies=[Depends(get_current_user)])
def update_experience(exp_id: int, exp: schemas.ExperienceCreate, db: Session = Depends(get_db)):
    return experience_crud.update(db, exp_id, exp, exclude_dates=["start_date", "end_date"])

@router.delete("/experiences/{exp_id}", dependencies=[Depends(get_current_user)])
def delete_experience(exp_id: int, db: Session = Depends(get_db)):
    return experience_crud.delete(db, exp_id)

# ── Education ─────────────────────────────────────────────────────────────────
@router.get("/education", response_model=List[schemas.EducationResponse])
@cached(ttl=300)
def get_education(db: Session = Depends(get_db)):
    return education_crud.get_all(db)

@router.post("/education", response_model=schemas.EducationResponse, dependencies=[Depends(get_current_user)])
def create_education(edu: schemas.EducationCreate, db: Session = Depends(get_db)):
    return education_crud.create(db, edu, date_fields=["start_date", "end_date"])

@router.put("/education/{edu_id}", response_model=schemas.EducationResponse, dependencies=[Depends(get_current_user)])
def update_education(edu_id: int, edu: schemas.EducationCreate, db: Session = Depends(get_db)):
    return education_crud.update(db, edu_id, edu, exclude_dates=["start_date", "end_date"])

@router.delete("/education/{edu_id}", dependencies=[Depends(get_current_user)])
def delete_education(edu_id: int, db: Session = Depends(get_db)):
    return education_crud.delete(db, edu_id)

# ── Services ──────────────────────────────────────────────────────────────────
@router.get("/services", response_model=List[schemas.ServiceResponse])
@cached(ttl=300)
def get_services(db: Session = Depends(get_db)):
    return service_crud.get_all(db)

@router.post("/services", response_model=schemas.ServiceResponse, dependencies=[Depends(get_current_user)])
def create_service(service: schemas.ServiceCreate, db: Session = Depends(get_db)):
    return service_crud.create(db, service)

@router.put("/services/{service_id}", response_model=schemas.ServiceResponse, dependencies=[Depends(get_current_user)])
def update_service(service_id: int, service: schemas.ServiceCreate, db: Session = Depends(get_db)):
    return service_crud.update(db, service_id, service)

@router.delete("/services/{service_id}", dependencies=[Depends(get_current_user)])
def delete_service(service_id: int, db: Session = Depends(get_db)):
    return service_crud.delete(db, service_id)

# ── Articles ──────────────────────────────────────────────────────────────────
@router.get("/articles", response_model=List[schemas.ArticleResponse])
@cached(ttl=120)
def get_articles(db: Session = Depends(get_db)):
    return article_crud.get_all(db)

@router.get("/articles/{slug}", response_model=schemas.ArticleResponse)
def get_article(slug: str, db: Session = Depends(get_db)):
    article = db.query(models.Article).filter(models.Article.slug == slug).first()
    if not article:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Article not found")
    return article

@router.post("/articles", response_model=schemas.ArticleResponse, dependencies=[Depends(get_current_user)])
def create_article(article: schemas.ArticleCreate, db: Session = Depends(get_db)):
    return article_crud.create(db, article, date_fields=["published_date"])

@router.delete("/articles/{article_id}", dependencies=[Depends(get_current_user)])
def delete_article(article_id: int, db: Session = Depends(get_db)):
    return article_crud.delete(db, article_id)
