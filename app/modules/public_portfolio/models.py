from sqlalchemy import Column, Integer, String, Text, Boolean, Date, ForeignKey
from app.core.database import Base

class Project(Base):
    __tablename__ = "public_projects"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    tech_stack = Column(String) # Comma separated
    image_url = Column(String, nullable=True)
    github_url = Column(String, nullable=True)
    live_url = Column(String, nullable=True)
    is_featured = Column(Boolean, default=False)
    gradient = Column(String, nullable=True)
    content = Column(Text, nullable=True)

class Skill(Base):
    __tablename__ = "public_skills"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    category = Column(String) # e.g., Frontend, Backend, DevOps
    proficiency = Column(Integer) # 1-100

class Service(Base):
    __tablename__ = "public_services"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(Text)
    icon = Column(String, nullable=True)
    features = Column(Text, nullable=True) # Comma separated

class Experience(Base):
    __tablename__ = "public_experiences"
    id = Column(Integer, primary_key=True, index=True)
    company = Column(String)
    role = Column(String)
    start_date = Column(String) # Store as YYYY-MM or similar
    end_date = Column(String, nullable=True)
    description = Column(Text)

class Education(Base):
    __tablename__ = "public_education"
    id = Column(Integer, primary_key=True, index=True)
    institution = Column(String)
    degree = Column(String)

class Article(Base):
    __tablename__ = "public_articles"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    slug = Column(String, unique=True, index=True)
    content = Column(Text)
    summary = Column(Text)
    category = Column(String)
    image_url = Column(String, nullable=True)
    published_date = Column(Date)
    is_published = Column(Boolean, default=True)
