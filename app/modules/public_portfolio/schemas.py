from pydantic import BaseModel
from typing import Optional, List

class ProjectBase(BaseModel):
    title: str
    description: str
    tech_stack: str
    image_url: Optional[str] = None
    github_url: Optional[str] = None
    live_url: Optional[str] = None
    is_featured: bool = False
    gradient: Optional[str] = None
    content: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectResponse(ProjectBase):
    id: int

    class Config:
        from_attributes = True

# --- Skill Schemas ---
class SkillBase(BaseModel):
    name: str
    category: str
    proficiency: int

class SkillCreate(SkillBase):
    pass

class SkillResponse(SkillBase):
    id: int
    class Config:
        from_attributes = True

# --- Experience Schemas ---
class ExperienceBase(BaseModel):
    company: str
    role: str
    start_date: str
    end_date: Optional[str] = None
    description: str

class ExperienceCreate(ExperienceBase):
    pass

class ExperienceResponse(ExperienceBase):
    id: int
    class Config:
        from_attributes = True

# --- Education Schemas ---
class EducationBase(BaseModel):
    institution: str
    degree: str

class EducationCreate(EducationBase):
    pass

class EducationResponse(EducationBase):
    id: int
    class Config:
        from_attributes = True

# --- Service Schemas ---
class ServiceBase(BaseModel):
    title: str
    description: str
    icon: Optional[str] = None
    features: Optional[str] = None

class ServiceCreate(ServiceBase):
    pass

class ServiceResponse(ServiceBase):
    id: int
    class Config:
        from_attributes = True

# --- Article Schemas ---
class ArticleBase(BaseModel):
    title: str
    slug: str
    content: str
    summary: str
    category: str
    image_url: Optional[str] = None
    published_date: str
    is_published: bool = True

class ArticleCreate(ArticleBase):
    pass

class ArticleResponse(ArticleBase):
    id: int
    class Config:
        from_attributes = True
