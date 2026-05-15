from typing import Type, TypeVar, List, Optional, Generic
from fastapi import HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.core.cache import invalidate_cache

ModelType = TypeVar("ModelType")
CreateSchema = TypeVar("CreateSchema", bound=BaseModel)

class CRUDBase(Generic[ModelType, CreateSchema]):
    def __init__(self, model: Type[ModelType], cache_prefix: str = ""):
        self.model = model
        self.cache_prefix = cache_prefix or model.__tablename__

    def get_all(self, db: Session) -> List[ModelType]:
        return db.query(self.model).all()

    def get_by_id(self, db: Session, id: int) -> Optional[ModelType]:
        obj = db.query(self.model).filter(self.model.id == id).first()
        if not obj:
            raise HTTPException(status_code=404, detail=f"{self.model.__name__} not found")
        return obj

    def create(self, db: Session, schema: CreateSchema, date_fields: Optional[List[str]] = None) -> ModelType:
        data = schema.model_dump()
        if date_fields:
            from datetime import datetime
            for field in date_fields:
                if data.get(field):
                    data[field] = datetime.strptime(data[field], "%Y-%m-%d").date()
                elif data.get(field) is None:
                    data[field] = None
        obj = self.model(**data)
        db.add(obj)
        db.commit()
        db.refresh(obj)
        invalidate_cache(self.cache_prefix)
        return obj

    def update(self, db: Session, id: int, schema: CreateSchema, exclude_dates: Optional[List[str]] = None) -> ModelType:
        obj = self.get_by_id(db, id)
        if not obj:
            raise HTTPException(status_code=404, detail=f"{self.model.__name__} not found")
        data = schema.model_dump()
        if exclude_dates:
            from datetime import datetime
            for field in exclude_dates:
                if data.get(field):
                    data[field] = datetime.strptime(data[field], "%Y-%m-%d").date()
                elif data.get(field) is None:
                    data[field] = None
        for key, value in data.items():
            setattr(obj, key, value)
        db.commit()
        db.refresh(obj)
        invalidate_cache(self.cache_prefix)
        return obj

    def delete(self, db: Session, id: int) -> dict:
        obj = self.get_by_id(db, id)
        if not obj:
            raise HTTPException(status_code=404, detail=f"{self.model.__name__} not found")
        db.delete(obj)
        db.commit()
        invalidate_cache(self.cache_prefix)
        return {"message": f"{self.model.__name__} deleted"}
