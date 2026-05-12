from sqlalchemy import Column, Integer, String, Text, Date, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base
from datetime import datetime

class BlogPost(Base):
    __tablename__ = "blog_posts"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    slug = Column(String, unique=True, index=True)
    content = Column(Text)
    summary = Column(Text, nullable=True)
    published = Column(Boolean, default=False)
    created_at = Column(Date, default=datetime.utcnow)
    
    # In a real app we might have a many-to-many relationship with tags.
    # For simplicity here we store as comma separated or string.
    tags = Column(String, nullable=True)
