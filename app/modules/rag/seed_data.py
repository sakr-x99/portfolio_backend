"""
Seed Script — Populates the database with initial portfolio data.
Run this once to populate skills, services, and other static data.

Usage:
    python -m app.modules.rag.seed_data
"""
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.core.database import SessionLocal, engine, Base
from app.modules.public_portfolio.models import Skill, Service, Experience, Education, Project

logger = logging.getLogger(__name__)

# Skills data
SKILLS_DATA = [
    # Backend Development
    {"name": "Python", "category": "Backend Development", "proficiency": 95},
    {"name": "FastAPI", "category": "Backend Development", "proficiency": 90},
    {"name": "PostgreSQL", "category": "Backend Development", "proficiency": 90},
    {"name": "REST APIs", "category": "Backend Development", "proficiency": 95},
    {"name": "Django", "category": "Backend Development", "proficiency": 85},

    # Frontend Development
    {"name": "React.js", "category": "Frontend Development", "proficiency": 80},
    {"name": "Next.js", "category": "Frontend Development", "proficiency": 75},
    {"name": "TypeScript", "category": "Frontend Development", "proficiency": 75},
    {"name": "JavaScript", "category": "Frontend Development", "proficiency": 80},
    {"name": "TailwindCSS", "category": "Frontend Development", "proficiency": 85},
    {"name": "Framer Motion", "category": "Frontend Development", "proficiency": 70},

    # DevOps & Tools
    {"name": "Docker", "category": "DevOps & Tools", "proficiency": 85},
    {"name": "Git", "category": "DevOps & Tools", "proficiency": 90},
    {"name": "Linux", "category": "DevOps & Tools", "proficiency": 80},
    {"name": "AWS", "category": "DevOps & Tools", "proficiency": 75},

    # AI & Machine Learning
    {"name": "OpenAI", "category": "AI & Machine Learning", "proficiency": 85},
    {"name": "TensorFlow", "category": "AI & Machine Learning", "proficiency": 75},
    {"name": "PyTorch", "category": "AI & Machine Learning", "proficiency": 75},
    {"name": "Machine Learning", "category": "AI & Machine Learning", "proficiency": 80},
    {"name": "Deep Learning", "category": "AI & Machine Learning", "proficiency": 75},

    # Strategy & Operations
    {"name": "Business Thinking", "category": "Strategy & Operations", "proficiency": 85},
]

# Default services
SERVICES_DATA = [
    {
        "title": "Backend Development",
        "description": "Building scalable, high-performance APIs and backend systems using Python, FastAPI, and PostgreSQL.",
        "icon": "server",
        "features": "REST APIs, GraphQL, Microservices, Database Design, Authentication",
    },
    {
        "title": "AI Integration",
        "description": "Integrating AI and Machine Learning into real-world products, RAG systems, and intelligent assistants.",
        "icon": "brain",
        "features": "RAG Systems, LLM Integration, Chatbots, AI Agents, Prompt Engineering",
    },
    {
        "title": "DevOps & Infrastructure",
        "description": "Setting up reliable, production-ready infrastructures with Docker, CI/CD, and cloud platforms.",
        "icon": "cloud",
        "features": "Docker, AWS, CI/CD Pipelines, Monitoring, Deployment Automation",
    },
    {
        "title": "Full-Stack Development",
        "description": "Building complete products from idea to deployment with modern frontend and backend technologies.",
        "icon": "code",
        "features": "React, Next.js, TypeScript, FastAPI, PostgreSQL, TailwindCSS",
    },
]


def seed_skills(db):
    """Populate skills table if empty."""
    existing = db.query(Skill).count()
    if existing > 0:
        logger.info(f"Skills table already has {existing} entries. Skipping.")
        return 0

    skills = [Skill(**data) for data in SKILLS_DATA]
    db.add_all(skills)
    db.commit()
    logger.info(f"Seeded {len(skills)} skills.")
    return len(skills)


def seed_services(db):
    """Populate services table if empty."""
    existing = db.query(Service).count()
    if existing > 0:
        logger.info(f"Services table already has {existing} entries. Skipping.")
        return 0

    services = [Service(**data) for data in SERVICES_DATA]
    db.add_all(services)
    db.commit()
    logger.info(f"Seeded {len(services)} services.")
    return len(services)


def seed_all():
    """Run all seeders."""
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        skills_count = seed_skills(db)
        services_count = seed_services(db)

        print(f"✅ Seed complete: {skills_count} skills, {services_count} services added.")
        print("🔁 Now run: POST /api/v1/rag/index to rebuild the knowledge base.")
    except Exception as e:
        db.rollback()
        logger.error(f"Seed failed: {e}", exc_info=True)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    seed_all()
