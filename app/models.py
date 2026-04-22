from sqlalchemy import Column, BigInteger, String, Text, Integer, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    phone = Column(String(20), unique=True, nullable=False, index=True)
    password_hash = Column(Text, nullable=False)
    role = Column(String(20), nullable=False, default="student")
    status = Column(String(20), nullable=False, default="active")
    created_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())
    updated_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())

    sessions = relationship("Session", back_populates="user")


class Case(Base):
    __tablename__ = "cases"

    id = Column(BigInteger, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    department = Column(String(100), nullable=False)
    difficulty = Column(String(50))
    patient_prompt = Column(Text, nullable=False)
    opening_message = Column(Text, nullable=False)
    expected_points = Column(Text)
    status = Column(String(20), nullable=False, default="active")
    created_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())
    updated_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())

    sessions = relationship("Session", back_populates="case")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(BigInteger, primary_key=True, index=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    case_id = Column(BigInteger, ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False)
    status = Column(String(20), nullable=False, default="in_progress")
    started_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())
    ended_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())
    updated_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())

    user = relationship("User", back_populates="sessions")
    case = relationship("Case", back_populates="sessions")
    messages = relationship("Message", back_populates="session")
    report = relationship("Report", back_populates="session", uselist=False)
    evaluation = relationship("Evaluation", back_populates="session", uselist=False)


class Message(Base):
    __tablename__ = "messages"

    id = Column(BigInteger, primary_key=True, index=True)
    session_id = Column(BigInteger, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    sender_type = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    sequence_no = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())

    session = relationship("Session", back_populates="messages")


class Report(Base):
    __tablename__ = "reports"

    id = Column(BigInteger, primary_key=True, index=True)
    session_id = Column(BigInteger, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, unique=True)
    chief_complaint = Column(Text)
    history_present_illness = Column(Text)
    related_history = Column(Text)
    preliminary_assessment = Column(Text)
    created_by = Column(String(20), nullable=False, default="ai")
    created_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())
    updated_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())

    session = relationship("Session", back_populates="report")


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(BigInteger, primary_key=True, index=True)
    session_id = Column(BigInteger, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, unique=True)
    communication_score = Column(Integer)
    completeness_score = Column(Integer)
    reasoning_score = Column(Integer)
    total_score = Column(Integer)
    feedback_text = Column(Text)
    created_by = Column(String(20), nullable=False, default="ai")
    created_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())

    session = relationship("Session", back_populates="evaluation")