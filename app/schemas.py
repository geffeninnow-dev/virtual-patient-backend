from pydantic import BaseModel, Field
from typing import Optional, List


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
    opening_message: Optional[OpeningMessage] = None


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


class StudentSubmission(BaseModel):
    session_id: int
    initial_judgment: str
    colposcopy_decision: str
    judgment_basis: str
    next_step_advice: Optional[str] = ""


class DimensionScore(BaseModel):
    name: str
    score: int
    max_score: int
    comment: str


class StructuredConsultationReport(BaseModel):
    chief_complaint: str
    history_of_present_illness: str
    menstrual_marital_reproductive_history: str
    past_history: str
    gynecological_history: str
    screening_and_examination_history: str
    student_initial_judgment: str
    student_colposcopy_decision: str
    student_judgment_basis: str
    student_next_step_advice: Optional[str] = ""
    system_reference_judgment: str


class ConsultationReportResponse(BaseModel):
    success: bool
    session_id: int
    score: int
    grade: str
    structured_report: StructuredConsultationReport
    dimension_scores: List[DimensionScore]
    overall_feedback: str
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    missed_key_points: List[str]