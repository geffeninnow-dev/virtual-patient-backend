from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import schemas, crud
from app.database import get_db
from app.ai_service import generate_patient_reply

router = APIRouter(prefix="/api", tags=["consultation"])


@router.post("/consultation/start", response_model=schemas.StartConsultationResponse)
def start_consultation(payload: schemas.StartConsultationRequest, db: Session = Depends(get_db)):
    case = crud.get_case_by_id(db, payload.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="病例不存在")

    session = crud.create_session(db, payload.user_id, payload.case_id)

    opening_message = crud.create_message(
        db=db,
        session_id=session.id,
        sender_type="ai_patient",
        content=case.opening_message,
    )

    return schemas.StartConsultationResponse(
        success=True,
        session_id=session.id,
        case={
            "id": case.id,
            "title": case.title,
            "department": case.department,
            "difficulty": case.difficulty,
        },
        opening_message=schemas.OpeningMessage(
            sender_type=opening_message.sender_type,
            content=opening_message.content,
        ),
    )


@router.post("/consultation/message", response_model=schemas.SendMessageResponse)
def send_message(payload: schemas.SendMessageRequest, db: Session = Depends(get_db)):
    session = crud.get_session_with_case(db, payload.session_id)

    if not session:
        raise HTTPException(status_code=404, detail="问诊会话不存在")

    if session.user_id != payload.user_id:
        raise HTTPException(status_code=403, detail="无权访问该问诊会话")

    crud.create_message(
        db=db,
        session_id=payload.session_id,
        sender_type="student",
        content=payload.message,
    )

    history_messages = crud.get_messages_by_session(db, payload.session_id)
    history_payload = [
        {"sender_type": m.sender_type, "content": m.content}
        for m in history_messages
    ]

    ai_reply = generate_patient_reply(
        patient_prompt=session.case.patient_prompt,
        history_messages=history_payload,
        user_message=payload.message,
    )

    reply_message = crud.create_message(
        db=db,
        session_id=payload.session_id,
        sender_type="ai_patient",
        content=ai_reply,
    )

    return schemas.SendMessageResponse(
        success=True,
        reply={
            "sender_type": reply_message.sender_type,
            "content": reply_message.content,
        },
    )


@router.post("/consultation/end", response_model=schemas.EndConsultationResponse)
def end_consultation(payload: schemas.EndConsultationRequest, db: Session = Depends(get_db)):
    session = crud.get_session_with_case(db, payload.session_id)

    if not session:
        raise HTTPException(status_code=404, detail="问诊会话不存在")

    if session.user_id != payload.user_id:
        raise HTTPException(status_code=403, detail="无权结束该问诊会话")

    ended_session = crud.end_session(db, payload.session_id)

    return schemas.EndConsultationResponse(
        success=True,
        message="问诊已结束",
        session_id=ended_session.id,
        status=ended_session.status,
    )