from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base

class DocCategory(Base):
    __tablename__ = "docs_categories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True) # e.g., "Next.js", "Docker"
    description = Column(String, nullable=True)
    
    tutorials = relationship("Tutorial", back_populates="category")

class Tutorial(Base):
    __tablename__ = "docs_tutorials"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(Text)
    order = Column(Integer, default=0)
    category_id = Column(Integer, ForeignKey("docs_categories.id"))
    
    category = relationship("DocCategory", back_populates="tutorials")
