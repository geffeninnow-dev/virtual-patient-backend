from sqlalchemy.orm import Session
from sqlalchemy import func
from app import models


def get_user_by_phone(db: Session, phone: str):
    return db.query(models.User).filter(models.User.phone == phone).first()


def create_user(db: Session, name: str, phone: str, password_hash: str):
    user = models.User(
        name=name,
        phone=phone,
        password_hash=password_hash,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_case_by_id(db: Session, case_id: int):
    return db.query(models.Case).filter(models.Case.id == case_id).first()


def create_session(db: Session, user_id: int, case_id: int):
    session = models.Session(
        user_id=user_id,
        case_id=case_id,
        status="in_progress",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_next_sequence_no(db: Session, session_id: int) -> int:
    max_seq = (
        db.query(func.max(models.Message.sequence_no))
        .filter(models.Message.session_id == session_id)
        .scalar()
    )
    return 1 if max_seq is None else max_seq + 1


def create_message(db: Session, session_id: int, sender_type: str, content: str):
    message = models.Message(
        session_id=session_id,
        sender_type=sender_type,
        content=content,
        sequence_no=get_next_sequence_no(db, session_id),
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_session_with_case(db: Session, session_id: int):
    return db.query(models.Session).filter(models.Session.id == session_id).first()


def get_messages_by_session(db: Session, session_id: int):
    return (
        db.query(models.Message)
        .filter(models.Message.session_id == session_id)
        .order_by(models.Message.sequence_no.asc())
        .all()
    )


def end_session(db: Session, session_id: int):
    session = (
        db.query(models.Session)
        .filter(models.Session.id == session_id)
        .first()
    )
    if not session:
        return None

    session.status = "completed"
    session.ended_at = func.now()
    db.commit()
    db.refresh(session)
    return session