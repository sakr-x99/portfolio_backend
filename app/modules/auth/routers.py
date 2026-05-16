from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from app.core.auth import create_access_token, verify_password

router = APIRouter()

class LoginRequest(BaseModel):
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):
    if not verify_password(request.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password")
    token = create_access_token(data={"sub": "admin"})
    return LoginResponse(access_token=token)
