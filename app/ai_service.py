from typing import List, Dict
import httpx

from app.config import settings


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
    if not settings.ai_api_key:
        return "【系统提示】AI_API_KEY 未配置，当前无法调用真实大模型。"

    if not settings.ai_base_url:
        return "【系统提示】AI_BASE_URL 未配置，当前无法调用真实大模型。"

    if not settings.ai_model_name:
        return "【系统提示】AI_MODEL_NAME 未配置，当前无法调用真实大模型。"

    messages = build_messages(
        patient_prompt=patient_prompt,
        history_messages=history_messages,
        user_message=user_message,
    )

    url = f"{settings.ai_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.ai_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": settings.ai_model_name,
        "messages": messages,
        "temperature": 0.6,
        "stream": False
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"].strip()

        if not content:
            return "我一时说不太清，您可以再问我具体一点吗？"

        return content

    except httpx.HTTPStatusError as e:
        try:
            error_text = e.response.text
        except Exception:
            error_text = str(e)
        return f"【模型调用失败】HTTP {e.response.status_code}: {error_text}"

    except Exception as e:
        return f"【模型调用异常】{str(e)}"