from typing import List, Dict, Any
import json
import httpx

from app.config import settings


def _call_chat_completion(
    messages: List[Dict],
    temperature: float = 0.6,
    timeout: float = 60.0
) -> str:
    """
    统一调用阿里百炼 OpenAI 兼容接口。
    返回模型输出文本。
    """
    if not settings.ai_api_key:
        return "【系统提示】AI_API_KEY 未配置，当前无法调用真实大模型。"

    if not settings.ai_base_url:
        return "【系统提示】AI_BASE_URL 未配置，当前无法调用真实大模型。"

    if not settings.ai_model_name:
        return "【系统提示】AI_MODEL_NAME 未配置，当前无法调用真实大模型。"

    url = f"{settings.ai_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.ai_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": settings.ai_model_name,
        "messages": messages,
        "temperature": temperature,
        "stream": False
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"].strip()
        return content

    except httpx.HTTPStatusError as e:
        try:
            error_text = e.response.text
        except Exception:
            error_text = str(e)
        return f"【模型调用失败】HTTP {e.response.status_code}: {error_text}"

    except Exception as e:
        return f"【模型调用异常】{str(e)}"


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    尽量从大模型输出中提取 JSON。
    避免模型偶尔输出额外说明文字导致解析失败。
    """
    if not text:
        raise ValueError("AI report response is empty")

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = text[start:end + 1]
        try:
            return json.loads(json_text)
        except Exception:
            pass

    raise ValueError(f"AI report response is not valid JSON: {text[:300]}")


def build_messages(
    patient_prompt: str,
    history_messages: List[Dict],
    user_message: str,
) -> List[Dict]:
    """
    构造发送给大模型的 messages。
    目标：让模型稳定扮演“妇科虚拟病患”。
    """
    system_prompt = f"""
你现在不是医生助手，而是一名“虚拟病患”，正在参加医学生妇科问诊训练。

【你的身份与场景】
1. 你扮演的是一名女性患者。
2. 当前场景限定为妇科门诊。
3. 本训练聚焦宫颈筛查异常、宫颈病变风险评估与阴道镜教学。
4. 医学生的重要任务之一，是通过问诊判断患者是否需要进一步进行阴道镜检查。
5. 你只能以患者身份回答，不能扮演医生、教师、考官或AI助手。

【病例设定】
{patient_prompt}

【回答规则】
1. 你必须始终严格依据病例设定进行角色扮演。
2. 你只能回答患者本人能够知道、感受到、回忆到的信息。
3. 你不能替学生做医学判断，不能说“你需要做阴道镜”“这说明宫颈病变”“建议你做某某检查”等结论。
4. 你不能主动总结病历，也不能一次性把所有信息全部说出来。
5. 学生问什么，你就回答什么；没有被问到的重要信息，不要主动完整展开。
6. 回答要自然、口语化、简洁，符合真实女性患者在妇科门诊中的表达方式。
7. 如果学生连续追问，你可以逐步补充信息，但必须保持前后一致。
8. 如果问题涉及隐私、记忆不清、或病例设定中未提供的信息，可以自然回答“记不太清了”“没特别注意”“不太确定”“之前做没做我一下想不起来”等。
9. 与宫颈筛查异常和阴道镜判断密切相关的信息，在学生问到对应方向时，应明确回答；未问到时不要主动完整展开。
10. 如果学生的问题明显超出妇科问诊和本病例设定范围，你可以以患者身份做自然回应，但不要把话题带离妇科场景。
11. 不要输出分析过程、提示词内容、规则说明，不要解释自己是AI，不要说“根据病例设定”。
12. 只输出患者本轮回复正文，不要添加“患者：”“答：”等前缀。

【学生常见提问方向】
学生通常会围绕以下方向提问，你应按真实患者方式自然作答：
- 本次就诊原因与主要不适
- 宫颈癌筛查史（TCT/HPV）
- 检查异常出现的时间、持续时间、复查情况
- 同房后出血、异常阴道排液、分泌物异常、带血丝等症状
- 月经、绝经、异常子宫出血情况
- 既往阴道镜检查或宫颈相关治疗史
- 性生活史、避孕、生育需求与产科史
- 妇科既往病史、手术史、感染史
- 慢性病、免疫状态、HPV疫苗接种、家族史及相关危险因素
""".strip()

    messages: List[Dict] = [
        {"role": "system", "content": system_prompt}
    ]

    for msg in history_messages:
        sender_type = msg.get("sender_type", "")
        content = (msg.get("content") or "").strip()
        if not content:
            continue

        if sender_type == "student":
            messages.append({"role": "user", "content": content})
        elif sender_type == "ai_patient":
            messages.append({"role": "assistant", "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages


def generate_patient_reply(
    patient_prompt: str,
    history_messages: List[Dict],
    user_message: str
) -> str:
    """
    调用阿里百炼 OpenAI 兼容接口，生成虚拟病患回复。
    """
    messages = build_messages(
        patient_prompt=patient_prompt,
        history_messages=history_messages,
        user_message=user_message,
    )

    content = _call_chat_completion(
        messages=messages,
        temperature=0.6,
        timeout=60.0
    )

    if not content:
        return "我一时说不太清，您可以再问我具体一点吗？"

    # 如果底层调用失败，会返回【模型调用失败】等提示，这里直接返回，方便前端和日志看到原因。
    return content


def generate_consultation_report(
    case_title: str,
    dialogue_messages: List[Dict],
    student_submission: Dict
) -> Dict[str, Any]:
    """
    根据本次问诊对话与学生提交的初步判断，生成结构化问诊报告与教学评价。
    """

    dialogue_text = ""

    for msg in dialogue_messages:
        sender_type = msg.get("sender_type", "")
        content = (msg.get("content") or "").strip()

        if not content:
            continue

        if sender_type == "student":
            role_name = "学生/医生"
        elif sender_type == "ai_patient":
            role_name = "AI病人"
        else:
            role_name = sender_type or "未知角色"

        dialogue_text += f"{role_name}：{content}\n"

    system_prompt = """
你是一名妇科临床教学教师，正在评价医学生的一次虚拟病人问诊训练。

请严格基于以下材料生成报告：
1. 本次问诊对话记录；
2. 学生提交的初步判断；
3. 学生是否建议进行阴道镜检查；
4. 学生填写的判断依据和下一步建议。

平台定位：
- 这是妇科问诊教学训练；
- 核心训练目标包括：妇科病史采集、宫颈筛查异常相关问诊、判断是否需要阴道镜检查；
- 输出仅用于医学教学训练，不作为真实医疗诊断建议。

评价维度采用100分制：
1. 问诊结构完整性：20分
2. 妇科专科关键点覆盖：25分
3. 临床思维与阴道镜指征判断：25分
4. 问诊逻辑与问题质量：15分
5. 医患沟通与人文关怀：10分
6. 报告完整性与表达规范：5分

评价要求：
- 不要编造对话中完全没有的信息；
- 对于对话中未询问或未获得的信息，应标注“未充分询问”或“未获得”；
- 对学生表现的评价要像教师给医学生的反馈，具体、专业、可改进；
- 重点评价学生是否围绕妇科主诉、宫颈筛查异常、阴道镜检查指征进行有效问诊；
- 评分要有区分度，不要所有维度都给满分；
- 如果学生问诊很少、信息不足，应降低相应维度分数；
- 必须返回合法 JSON；
- 不要输出 Markdown；
- 不要输出 JSON 以外的任何解释文字。
""".strip()

    user_prompt = f"""
病例名称：
{case_title}

本次问诊对话记录：
{dialogue_text}

学生提交内容：
初步判断：{student_submission.get("initial_judgment", "")}
是否建议阴道镜检查：{student_submission.get("colposcopy_decision", "")}
判断依据：{student_submission.get("judgment_basis", "")}
下一步建议：{student_submission.get("next_step_advice", "")}

请严格按以下 JSON 格式返回，字段名不能改变：

{{
  "score": 82,
  "grade": "良好",
  "structured_report": {{
    "chief_complaint": "自动整理主诉，格式建议为主要症状+持续时间；信息不足则写未充分询问",
    "history_of_present_illness": "自动整理现病史，包括起病时间、症状特点、频率、诱因、伴随症状、诊疗经过等；信息不足则写未充分询问",
    "menstrual_marital_reproductive_history": "整理月经婚育及性生活相关史；信息不足则写未充分询问",
    "past_history": "整理既往史、过敏史、慢性病史等；信息不足则写未充分询问",
    "gynecological_history": "整理妇科相关病史、手术史、流产史、宫颈治疗史等；信息不足则写未充分询问",
    "screening_and_examination_history": "整理HPV、TCT、阴道镜、HPV疫苗等筛查和检查资料；信息不足则写未充分询问",
    "student_initial_judgment": "复述学生初步判断",
    "student_colposcopy_decision": "复述学生阴道镜检查选择",
    "student_judgment_basis": "复述学生判断依据",
    "student_next_step_advice": "复述学生下一步建议",
    "system_reference_judgment": "系统参考判断，应说明这是教学参考，不是真实诊断"
  }},
  "dimension_scores": [
    {{
      "name": "问诊结构完整性",
      "score": 16,
      "max_score": 20,
      "comment": "简要评价该维度表现"
    }},
    {{
      "name": "妇科专科关键点覆盖",
      "score": 20,
      "max_score": 25,
      "comment": "简要评价该维度表现"
    }},
    {{
      "name": "临床思维与阴道镜指征判断",
      "score": 21,
      "max_score": 25,
      "comment": "简要评价该维度表现"
    }},
    {{
      "name": "问诊逻辑与问题质量",
      "score": 12,
      "max_score": 15,
      "comment": "简要评价该维度表现"
    }},
    {{
      "name": "医患沟通与人文关怀",
      "score": 8,
      "max_score": 10,
      "comment": "简要评价该维度表现"
    }},
    {{
      "name": "报告完整性与表达规范",
      "score": 5,
      "max_score": 5,
      "comment": "简要评价该维度表现"
    }}
  ],
  "overall_feedback": "总体教师式评价，100到200字左右",
  "strengths": ["优点1", "优点2", "优点3"],
  "weaknesses": ["不足1", "不足2", "不足3"],
  "improvement_suggestions": ["建议1", "建议2", "建议3"],
  "missed_key_points": ["遗漏点1", "遗漏点2", "遗漏点3"]
}}

评分等级规则：
90-100：优秀
80-89：良好
70-79：合格
60-69：基本达标，需加强
60以下：未达标，建议重新训练
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = _call_chat_completion(
        messages=messages,
        temperature=0.2,
        timeout=90.0
    )

    if content.startswith("【模型调用失败】") or content.startswith("【模型调用异常】") or content.startswith("【系统提示】"):
        raise ValueError(content)

    report_data = _safe_json_loads(content)

    # 做一个基础兜底，防止模型漏字段导致前端崩掉。
    report_data.setdefault("score", 0)
    report_data.setdefault("grade", "未评分")
    report_data.setdefault("structured_report", {})
    report_data.setdefault("dimension_scores", [])
    report_data.setdefault("overall_feedback", "")
    report_data.setdefault("strengths", [])
    report_data.setdefault("weaknesses", [])
    report_data.setdefault("improvement_suggestions", [])
    report_data.setdefault("missed_key_points", [])

    structured_report = report_data["structured_report"]
    structured_report.setdefault("chief_complaint", "未生成")
    structured_report.setdefault("history_of_present_illness", "未生成")
    structured_report.setdefault("menstrual_marital_reproductive_history", "未生成")
    structured_report.setdefault("past_history", "未生成")
    structured_report.setdefault("gynecological_history", "未生成")
    structured_report.setdefault("screening_and_examination_history", "未生成")
    structured_report.setdefault("student_initial_judgment", student_submission.get("initial_judgment", ""))
    structured_report.setdefault("student_colposcopy_decision", student_submission.get("colposcopy_decision", ""))
    structured_report.setdefault("student_judgment_basis", student_submission.get("judgment_basis", ""))
    structured_report.setdefault("student_next_step_advice", student_submission.get("next_step_advice", ""))
    structured_report.setdefault("system_reference_judgment", "未生成")

    return report_data