from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app import schemas, crud
from app.auth import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/api", tags=["auth"])


@router.post("/register", response_model=schemas.RegisterResponse)
def register(payload: schemas.RegisterRequest, db: Session = Depends(get_db)):
    existing_user = crud.get_user_by_phone(db, payload.phone)
    if existing_user:
        return schemas.RegisterResponse(
            success=False,
            message="该手机号已注册"
        )

    user = crud.create_user(
        db=db,
        name=payload.name,
        phone=payload.phone,
        password_hash=hash_password(payload.password),
    )

    return schemas.RegisterResponse(
        success=True,
        message="注册成功",
        user=user,
    )


@router.post("/login", response_model=schemas.LoginResponse)
def login(payload: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = crud.get_user_by_phone(db, payload.phone)
    if not user:
        return schemas.LoginResponse(success=False, message="手机号或密码错误")

    if not verify_password(payload.password, user.password_hash):
        return schemas.LoginResponse(success=False, message="手机号或密码错误")

    token = create_access_token(subject=str(user.id))

    return schemas.LoginResponse(
        success=True,
        message="登录成功",
        token=token,
        user=user,
    )