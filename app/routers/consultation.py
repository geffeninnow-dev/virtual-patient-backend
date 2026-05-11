from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import schemas, crud
from app.database import get_db
from app.ai_service import generate_patient_reply, generate_consultation_report

router = APIRouter(prefix="/api", tags=["consultation"])


@router.post("/consultation/start", response_model=schemas.StartConsultationResponse)
def start_consultation(payload: schemas.StartConsultationRequest, db: Session = Depends(get_db)):
    """
    开始问诊训练。

    注意：
    现在流程改为“学生/医生先问，AI病人再回答”，
    因此这里只创建问诊会话，不再自动插入 AI 病人的开场白。
    """
    case = crud.get_case_by_id(db, payload.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="病例不存在")

    session = crud.create_session(db, payload.user_id, payload.case_id)

    return schemas.StartConsultationResponse(
        success=True,
        session_id=session.id,
        case={
            "id": case.id,
            "title": case.title,
            "department": case.department,
            "difficulty": case.difficulty,
        },
        opening_message=None,
    )


@router.post("/consultation/message", response_model=schemas.SendMessageResponse)
def send_message(payload: schemas.SendMessageRequest, db: Session = Depends(get_db)):
    """
    学生发送问诊问题后，AI虚拟病人根据病例设定和历史对话进行回答。
    """
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
    """
    结束问诊会话。
    前端结束后会弹出学生初步判断表单，再调用 /consultation/report 生成报告。
    """
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


@router.post("/consultation/report", response_model=schemas.ConsultationReportResponse)
def generate_report(payload: schemas.StudentSubmission, db: Session = Depends(get_db)):
    """
    生成本次问诊训练报告。

    流程：
    1. 读取本次问诊会话；
    2. 读取本次问诊全部对话记录；
    3. 合并学生提交的初步判断、阴道镜检查选择、判断依据和下一步建议；
    4. 调用大模型生成结构化问诊报告与教师式评价反馈。
    """
    session = crud.get_session_with_case(db, payload.session_id)

    if not session:
        raise HTTPException(status_code=404, detail="问诊会话不存在")

    messages = crud.get_messages_by_session(db, payload.session_id)

    if not messages:
        raise HTTPException(status_code=400, detail="当前问诊会话没有对话记录，无法生成报告")

    dialogue_messages = [
        {
            "sender_type": m.sender_type,
            "content": m.content,
        }
        for m in messages
    ]

    case_title = session.case.title if session.case else "妇科问诊训练"

    student_submission = {
        "initial_judgment": payload.initial_judgment,
        "colposcopy_decision": payload.colposcopy_decision,
        "judgment_basis": payload.judgment_basis,
        "next_step_advice": payload.next_step_advice or "",
    }

    try:
        report_data = generate_consultation_report(
            case_title=case_title,
            dialogue_messages=dialogue_messages,
            student_submission=student_submission,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成报告失败：{str(e)}")

    return schemas.ConsultationReportResponse(
        success=True,
        session_id=payload.session_id,
        score=report_data.get("score", 0),
        grade=report_data.get("grade", "未评分"),
        structured_report=report_data.get("structured_report", {}),
        dimension_scores=report_data.get("dimension_scores", []),
        overall_feedback=report_data.get("overall_feedback", ""),
        strengths=report_data.get("strengths", []),
        weaknesses=report_data.get("weaknesses", []),
        improvement_suggestions=report_data.get("improvement_suggestions", []),
        missed_key_points=report_data.get("missed_key_points", []),
    )