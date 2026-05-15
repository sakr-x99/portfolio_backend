from app.core.database import SessionLocal
from app.modules.github_trends.models import TrendingRepo

db = SessionLocal()
try:
    count = db.query(TrendingRepo).count()
    print(f"Total TrendingRepos: {count}")
    if count > 0:
        repos = db.query(TrendingRepo).limit(5).all()
        for repo in repos:
            print(f"- {repo.full_name}: active={repo.is_active}, stars={repo.stars}")
finally:
    db.close()
