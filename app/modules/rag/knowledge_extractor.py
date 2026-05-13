"""
Knowledge Extractor
Fetches all portfolio data from PostgreSQL and generates structured Markdown files
optimized for RAG retrieval. Each section becomes a separate .md file.
"""
import os
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.modules.public_portfolio import models
from . import config


def _ensure_knowledge_dir():
    """Create the knowledge directory if it doesn't exist."""
    os.makedirs(config.KNOWLEDGE_DIR, exist_ok=True)


def extract_about(db: Session) -> str:
    """Generate about.md from aggregated profile data."""
    skills = db.query(models.Skill).all()
    experiences = db.query(models.Experience).all()
    services = db.query(models.Service).all()

    # Build a professional summary from real data
    skill_names = [s.name for s in skills]
    categories = list(set(s.category for s in skills))
    service_titles = [s.title for s in services]

    md = "# About Mohamed Sakr\n\n"
    md += "## Professional Summary\n\n"
    md += "Mohamed Sakr is a Backend-focused developer passionate about building scalable systems, "
    md += "modern APIs, and reliable infrastructures.\n\n"

    if categories:
        md += f"**Areas of Expertise:** {', '.join(categories)}\n\n"
    if skill_names:
        md += f"**Technical Skills:** {', '.join(skill_names)}\n\n"
    if service_titles:
        md += f"**Services Offered:** {', '.join(service_titles)}\n\n"
    if experiences:
        latest = experiences[0]
        md += f"**Current/Latest Role:** {latest.role} at {latest.company}\n\n"

    return md


def extract_skills(db: Session) -> str:
    """Generate skills.md grouped by category."""
    skills = db.query(models.Skill).all()
    if not skills:
        return "# Skills\n\nNo skills data available yet.\n"

    # Group by category
    categories: dict[str, list] = {}
    for skill in skills:
        cat = skill.category or "Other"
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(skill)

    md = "# Technical Skills\n\n"
    md += f"Mohamed Sakr has expertise in {len(skills)} technologies across {len(categories)} categories.\n\n"

    for cat, cat_skills in categories.items():
        md += f"## {cat}\n\n"
        for s in cat_skills:
            proficiency = s.proficiency or 0
            level = "Expert" if proficiency >= 80 else "Advanced" if proficiency >= 60 else "Intermediate" if proficiency >= 40 else "Beginner"
            md += f"- **{s.name}** — Proficiency: {proficiency}% ({level})\n"
        md += "\n"

    return md


def extract_experience(db: Session) -> str:
    """Generate experience.md with work history timeline."""
    experiences = db.query(models.Experience).order_by(models.Experience.start_date.desc()).all()
    if not experiences:
        return "# Work Experience\n\nNo experience data available yet.\n"

    md = "# Work Experience\n\n"
    md += f"Mohamed Sakr has {len(experiences)} professional experience entries.\n\n"

    for exp in experiences:
        end = exp.end_date.strftime("%B %Y") if exp.end_date else "Present"
        start = exp.start_date.strftime("%B %Y") if exp.start_date else "Unknown"
        md += f"## {exp.role} at {exp.company}\n\n"
        md += f"**Duration:** {start} — {end}\n\n"
        if exp.description:
            md += f"{exp.description}\n\n"
        md += "---\n\n"

    return md


def extract_education(db: Session) -> str:
    """Generate education.md with academic background."""
    education = db.query(models.Education).order_by(models.Education.start_date.desc()).all()
    if not education:
        return "# Education\n\nNo education data available yet.\n"

    md = "# Education\n\n"
    for edu in education:
        end = edu.end_date.strftime("%Y") if edu.end_date else "Present"
        start = edu.start_date.strftime("%Y") if edu.start_date else "Unknown"
        md += f"## {edu.degree}\n\n"
        md += f"**Institution:** {edu.institution}\n\n"
        md += f"**Period:** {start} — {end}\n\n"
        md += "---\n\n"

    return md


def extract_projects(db: Session) -> str:
    """Generate projects.md with full project details."""
    projects = db.query(models.Project).all()
    if not projects:
        return "# Projects\n\nNo projects data available yet.\n"

    md = "# Projects Portfolio\n\n"
    md += f"Mohamed Sakr has built {len(projects)} projects.\n\n"

    featured = [p for p in projects if p.is_featured]
    if featured:
        md += f"**Featured Projects:** {', '.join(p.title for p in featured)}\n\n"

    for proj in projects:
        md += f"## {proj.title}\n\n"
        if proj.is_featured:
            md += "⭐ **Featured Project**\n\n"
        if proj.description:
            md += f"{proj.description}\n\n"
        if proj.tech_stack:
            techs = [t.strip() for t in proj.tech_stack.split(",")]
            md += f"**Tech Stack:** {', '.join(techs)}\n\n"
        if proj.github_url:
            md += f"**GitHub:** {proj.github_url}\n\n"
        if proj.live_url:
            md += f"**Live Demo:** {proj.live_url}\n\n"
        
        # Include the full README/Content for deep RAG knowledge
        if proj.content:
            md += "### Project README / Documentation\n\n"
            md += f"{proj.content}\n\n"
            
        md += "---\n\n"

    return md


def extract_services(db: Session) -> str:
    """Generate services.md with all offered services."""
    services = db.query(models.Service).all()
    if not services:
        return "# Services\n\nNo services data available yet.\n"

    md = "# Services Offered\n\n"
    md += f"Mohamed Sakr offers {len(services)} professional services.\n\n"

    for svc in services:
        md += f"## {svc.title}\n\n"
        if svc.description:
            md += f"{svc.description}\n\n"
        if svc.features:
            features = [f.strip() for f in svc.features.split(",")]
            md += "**Core Technologies/Features:**\n\n"
            for feat in features:
                md += f"- {feat}\n"
            md += "\n"
        md += "---\n\n"

    return md


def extract_articles(db: Session) -> str:
    """Generate articles.md with blog content."""
    articles = db.query(models.Article).filter(models.Article.is_published == True).all()
    if not articles:
        return "# Articles & Blog\n\nNo articles published yet.\n"

    md = "# Articles & Insights\n\n"
    md += f"Mohamed Sakr has published {len(articles)} articles.\n\n"

    for article in articles:
        md += f"## {article.title}\n\n"
        if article.category:
            md += f"**Category:** {article.category}\n\n"
        if article.published_date:
            md += f"**Published:** {article.published_date.strftime('%B %d, %Y')}\n\n"
        if article.summary:
            md += f"**Summary:** {article.summary}\n\n"
        if article.content:
            # Include the full content for RAG indexing
            md += f"{article.content}\n\n"
        md += "---\n\n"

    return md


def extract_all() -> dict[str, str]:
    """
    Main extraction function.
    Connects to the database, extracts all portfolio data,
    and writes structured Markdown files to the knowledge directory.
    Returns a dict of {filename: content} for verification.
    """
    _ensure_knowledge_dir()

    db = SessionLocal()
    try:
        extractors = {
            "about.md": extract_about,
            "skills.md": extract_skills,
            "experience.md": extract_experience,
            "education.md": extract_education,
            "projects.md": extract_projects,
            "services.md": extract_services,
            "articles.md": extract_articles,
        }

        results = {}
        # Dynamic Extraction from DB
        for filename, extractor_fn in extractors.items():
            content = extractor_fn(db)
            filepath = os.path.join(config.KNOWLEDGE_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            results[filename] = content
            print(f"  ✓ Generated {filename} ({len(content)} chars)")

        # Copy static knowledge files
        static_dir = os.path.join(os.path.dirname(__file__), "knowledge")
        if os.path.exists(static_dir):
            for static_file in os.listdir(static_dir):
                if static_file.endswith(".md"):
                    with open(os.path.join(static_dir, static_file), "r", encoding="utf-8") as sf:
                        content = sf.read()
                    filepath = os.path.join(config.KNOWLEDGE_DIR, static_file)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)
                    results[static_file] = content
                    print(f"  ✓ Copied static {static_file} ({len(content)} chars)")

        return results

    finally:
        db.close()
