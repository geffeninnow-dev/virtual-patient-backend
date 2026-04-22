from pydantic import BaseModel, Field
from typing import Optional


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    phone: str = Field(..., min_length=6, max_length=20)
    password: str = Field(..., min_length=6, max_length=100)


class LoginRequest(BaseModel):
    phone: str
    password: str


class UserResponse(BaseModel):
    id: int
    name: str
    phone: str
    role: str

    class Config:
        from_attributes = True


class RegisterResponse(BaseModel):
    success: bool
    message: str
    user: Optional[UserResponse] = None


class LoginResponse(BaseModel):
    success: bool
    message: str
    token: Optional[str] = None
    user: Optional[UserResponse] = None


class StartConsultationRequest(BaseModel):
    user_id: int
    case_id: int


class OpeningMessage(BaseModel):
    sender_type: str
    content: str


class StartConsultationResponse(BaseModel):
    success: bool
    session_id: int
    case: dict
    opening_message: OpeningMessage


class SendMessageRequest(BaseModel):
    session_id: int
    user_id: int
    message: str


class SendMessageResponse(BaseModel):
    success: bool
    reply: dict


class EndConsultationRequest(BaseModel):
    session_id: int
    user_id: int


class EndConsultationResponse(BaseModel):
    success: bool
    message: str
    session_id: int
    status: str