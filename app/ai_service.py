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

def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _analyze_dialogue_coverage(dialogue_messages: List[Dict]) -> Dict[str, Any]:
    """
    分析本次对话中学生实际问到了哪些病史维度。
    注意：这里只分析学生/医生提问，不分析 AI 病人自己说了什么。
    """
    student_questions = []

    for msg in dialogue_messages:
        sender_type = msg.get("sender_type", "")
        content = (msg.get("content") or "").strip()

        if sender_type == "student" and content:
            student_questions.append(content)

    student_text = "\n".join(student_questions)

    coverage = {
        "student_turn_count": len(student_questions),

        # 主诉/就诊原因
        "chief_complaint": _contains_any(
            student_text,
            ["哪里不舒服", "哪儿不舒服", "怎么了", "什么不舒服", "为什么来", "就诊原因", "主要问题", "主要症状"]
        ),

        # 现病史：时间、程度、频率、诱因、伴随症状、处理经过等
        "history_of_present_illness": _contains_any(
            student_text,
            ["多久", "多长时间", "什么时候", "几次", "频率", "量多", "量少", "颜色", "疼不疼", "腹痛", "腰酸", "诱因", "缓解", "加重", "处理", "用药", "治疗", "伴随"]
        ),

        # 月经史
        "menstrual_history": _contains_any(
            student_text,
            ["月经", "经期", "周期", "末次月经", "LMP", "绝经", "停经", "经量", "痛经"]
        ),

        # 婚育史/性生活/避孕
        "marital_reproductive_sexual_history": _contains_any(
            student_text,
            ["结婚", "婚育", "生育", "怀孕", "妊娠", "生产", "流产", "剖宫产", "性生活", "同房", "性行为", "性伴侣", "避孕", "安全套", "不安全"]
        ),

        # 既往史、慢性病、过敏史
        "past_history": _contains_any(
            student_text,
            ["既往", "以前", "病史", "高血压", "糖尿病", "慢性病", "过敏", "手术", "用药史", "免疫", "长期服药"]
        ),

        # 妇科既往史、宫颈治疗史、阴道镜史
        "gynecological_history": _contains_any(
            student_text,
            ["妇科病", "宫颈炎", "宫颈病变", "宫颈治疗", "宫颈手术", "LEEP", "锥切", "阴道镜", "活检", "妇科检查"]
        ),

        # HPV/TCT/筛查/疫苗
        "screening_and_examination_history": _contains_any(
            student_text,
            ["HPV", "TCT", "筛查", "宫颈癌筛查", "细胞学", "疫苗", "HPV疫苗", "检查结果", "复查"]
        ),

        # 白带/分泌物
        "vaginal_discharge": _contains_any(
            student_text,
            ["白带", "分泌物", "异味", "气味", "颜色", "瘙痒", "外阴痒"]
        ),

        # 接触性出血/同房后出血
        "contact_bleeding": _contains_any(
            student_text,
            ["同房后出血", "接触性出血", "性生活后出血", "同房之后出血", "房事后出血"]
        ),
    }

    covered_count = sum(
        1 for key, value in coverage.items()
        if key != "student_turn_count" and value
    )

    coverage["covered_key_area_count"] = covered_count
    coverage["student_questions"] = student_questions

    return coverage


def _grade_from_score(score: int) -> str:
    if score >= 90:
        return "优秀"
    if score >= 80:
        return "良好"
    if score >= 70:
        return "合格"
    if score >= 60:
        return "基本达标，需加强"
    return "未达标，建议重新训练"


def _calculate_score_cap(coverage: Dict[str, Any]) -> int:
    """
    根据实际问诊轮次和关键维度覆盖情况设置最高分。
    目的：防止问诊很草率时模型仍然给高分。
    """
    student_turn_count = coverage.get("student_turn_count", 0)
    covered_key_area_count = coverage.get("covered_key_area_count", 0)

    # 先根据问诊轮次设置基础上限
    if student_turn_count <= 2:
        cap = 50
    elif student_turn_count <= 3:
        cap = 60
    elif student_turn_count <= 5:
        cap = 70
    elif student_turn_count <= 8:
        cap = 80
    else:
        cap = 90

    # 再根据关键维度覆盖情况限制上限
    if covered_key_area_count <= 2:
        cap = min(cap, 55)
    elif covered_key_area_count <= 4:
        cap = min(cap, 65)
    elif covered_key_area_count <= 6:
        cap = min(cap, 78)

    # 妇科问诊关键项缺失时进一步扣上限
    if not coverage.get("menstrual_history"):
        cap -= 5

    if not coverage.get("past_history"):
        cap -= 5

    if not coverage.get("screening_and_examination_history"):
        cap -= 8

    if not coverage.get("gynecological_history"):
        cap -= 5

    # 最低保留一个合理下限
    return max(35, cap)


def _apply_report_constraints(report_data: Dict[str, Any], coverage: Dict[str, Any]) -> Dict[str, Any]:
    """
    对模型生成结果进行后处理：
    1. 没问到的病史维度强制标注为“未涉及”；
    2. 分数超过上限时强制压低；
    3. 等级与分数重新匹配；
    4. 修复月经婚育史重复套娃问题。
    """
    structured_report = report_data.setdefault("structured_report", {})

    # 主诉：如果学生连就诊原因都没问，主诉也不应自动生成
    if not coverage.get("chief_complaint"):
        structured_report["chief_complaint"] = "未涉及"

    # 现病史：如果没有追问症状时间、频率、性质等，不能自动写完整现病史
    if not coverage.get("history_of_present_illness"):
        structured_report["history_of_present_illness"] = "未涉及"

    # 月经、婚育、性生活相关史：只保留本次问诊实际涉及的信息，避免重复套娃。
    existing_mmr = structured_report.get("menstrual_marital_reproductive_history", "")

    mmr_parts = []

    if coverage.get("menstrual_history"):
        if "月经史：" in existing_mmr:
            menstrual_text = existing_mmr.split("月经史：", 1)[1].split("；", 1)[0].strip()
            mmr_parts.append(f"月经史：{menstrual_text or '本次问诊已涉及'}")
        else:
            mmr_parts.append("月经史：本次问诊已涉及")
    else:
        mmr_parts.append("月经史：未涉及")

    if coverage.get("marital_reproductive_sexual_history"):
        sexual_text = ""

        if "性生活相关史：" in existing_mmr:
            sexual_text = existing_mmr.split("性生活相关史：", 1)[1].split("；", 1)[0].strip()
        elif "性生活/婚育相关史：" in existing_mmr:
            sexual_text = existing_mmr.split("性生活/婚育相关史：", 1)[1].split("；", 1)[0].strip()
        elif "婚育/性生活相关史：" in existing_mmr:
            sexual_text = existing_mmr.split("婚育/性生活相关史：", 1)[1].split("；", 1)[0].strip()

        mmr_parts.append(f"婚育/性生活相关史：{sexual_text or '本次问诊已涉及'}")
    else:
        mmr_parts.append("婚育/性生活相关史：未涉及")

    structured_report["menstrual_marital_reproductive_history"] = "；".join(mmr_parts)

    if not coverage.get("past_history"):
        structured_report["past_history"] = "未涉及"

    if not coverage.get("gynecological_history"):
        structured_report["gynecological_history"] = "未涉及"

    if not coverage.get("screening_and_examination_history"):
        structured_report["screening_and_examination_history"] = "未涉及"

    # 强制限制分数
    cap = _calculate_score_cap(coverage)
    raw_score = int(report_data.get("score", 0) or 0)
    final_score = min(raw_score, cap)

    report_data["score"] = final_score
    report_data["grade"] = _grade_from_score(final_score)

    # 如果总分被压低，同步压低维度分，避免维度分之和看起来仍然过高
    dimension_scores = report_data.get("dimension_scores", [])
    if isinstance(dimension_scores, list):
        total_dimension_score = 0

        for item in dimension_scores:
            try:
                total_dimension_score += int(item.get("score", 0))
            except Exception:
                pass

        if total_dimension_score > final_score and total_dimension_score > 0:
            ratio = final_score / total_dimension_score

            for item in dimension_scores:
                try:
                    original = int(item.get("score", 0))
                    item["score"] = max(0, int(round(original * ratio)))
                except Exception:
                    pass

    # 追加遗漏提醒，确保学生知道哪些没问
    missed_key_points = report_data.setdefault("missed_key_points", [])

    def add_missing(text: str):
        if text not in missed_key_points:
            missed_key_points.append(text)

    if not coverage.get("menstrual_history"):
        add_missing("未询问月经史，如月经周期、经期、经量、末次月经或是否绝经。")

    if not coverage.get("past_history"):
        add_missing("未询问既往史、慢性病史、过敏史或长期用药史。")

    if not coverage.get("gynecological_history"):
        add_missing("未询问既往妇科疾病史、宫颈治疗史或既往阴道镜检查史。")

    if not coverage.get("screening_and_examination_history"):
        add_missing("未询问HPV、TCT、宫颈癌筛查或HPV疫苗接种情况。")

    if not coverage.get("contact_bleeding"):
        add_missing("未明确追问是否存在同房后出血或接触性出血。")

    return report_data
    
def generate_consultation_report(
    case_title: str,
    dialogue_messages: List[Dict],
    student_submission: Dict
) -> Dict[str, Any]:
    """
    根据本次问诊对话与学生提交的初步判断，生成结构化问诊报告与教学评价。
    本函数严格限制模型只能基于本次对话生成报告，未问到的信息必须标注“未涉及”。
    """

    coverage = _analyze_dialogue_coverage(dialogue_messages)

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

    coverage_summary = f"""
本次问诊轮次统计：
- 学生/医生提问轮次：{coverage.get("student_turn_count", 0)}
- 实际覆盖关键维度数量：{coverage.get("covered_key_area_count", 0)}

本次问诊是否涉及以下维度：
- 主诉/就诊原因：{"已涉及" if coverage.get("chief_complaint") else "未涉及"}
- 现病史细节：{"已涉及" if coverage.get("history_of_present_illness") else "未涉及"}
- 月经史：{"已涉及" if coverage.get("menstrual_history") else "未涉及"}
- 婚育史/性生活/避孕相关史：{"已涉及" if coverage.get("marital_reproductive_sexual_history") else "未涉及"}
- 既往史/慢性病史/过敏史：{"已涉及" if coverage.get("past_history") else "未涉及"}
- 妇科既往史/宫颈治疗史/阴道镜史：{"已涉及" if coverage.get("gynecological_history") else "未涉及"}
- HPV/TCT/宫颈筛查/HPV疫苗：{"已涉及" if coverage.get("screening_and_examination_history") else "未涉及"}
- 白带/阴道分泌物：{"已涉及" if coverage.get("vaginal_discharge") else "未涉及"}
- 同房后出血/接触性出血：{"已涉及" if coverage.get("contact_bleeding") else "未涉及"}
""".strip()

    system_prompt = """
你是一名严格的妇科临床教学教师，正在评价医学生的一次虚拟病人问诊训练。

请严格基于“本次问诊对话记录”生成报告与评价。

最重要的规则：
1. 结构化问诊报告只能整理本次对话中明确出现的信息。
2. 学生没有问到、AI病人也没有在本次对话中回答的信息，必须写“未涉及”。
3. 绝对不能根据医学常识、病例名称、常规病历模板或你自己的推测补全信息。
4. 绝对不能编造月经史、婚育史、既往史、筛查史、检查结果、家族史等内容。
5. 如果某一项只问到一部分，要如实写出已问到部分，并明确标注其他部分“未涉及”。
6. 评分必须严格。问诊轮次少、关键病史遗漏多，不能给高分，更不能评为“良好”或“优秀”。
7. 本次输出仅用于医学教学训练，不作为真实医疗诊断建议。
8. 必须返回合法 JSON，不要输出 Markdown，不要输出 JSON 以外的任何解释文字。

评价维度采用100分制：
1. 问诊结构完整性：20分
2. 妇科专科关键点覆盖：25分
3. 临床思维与阴道镜指征判断：25分
4. 问诊逻辑与问题质量：15分
5. 医患沟通与人文关怀：10分
6. 报告完整性与表达规范：5分

评分原则：
- 如果学生提问少于3轮，总分通常不应超过60分；
- 如果未询问月经史、既往史、妇科既往史、HPV/TCT筛查史等关键内容，总分通常不应超过65分；
- 如果仅询问了主诉、白带和性生活安全性等少量问题，应明确指出问诊不完整；
- 对遗漏项要在 weaknesses、improvement_suggestions 和 missed_key_points 中明确指出。
""".strip()

    user_prompt = f"""
病例名称：
{case_title}

本次问诊对话记录：
{dialogue_text}

系统对本次对话覆盖情况的客观统计：
{coverage_summary}

学生提交内容：
初步判断：{student_submission.get("initial_judgment", "")}
是否建议阴道镜检查：{student_submission.get("colposcopy_decision", "")}
判断依据：{student_submission.get("judgment_basis", "")}
下一步建议：{student_submission.get("next_step_advice", "")}

请严格按以下 JSON 格式返回，字段名不能改变。
注意：下面只是字段格式，不代表可以照抄示例内容。所有内容必须来自本次对话；未问到的信息必须写“未涉及”。

{{
  "score": 0,
  "grade": "未评分",
  "structured_report": {{
    "chief_complaint": "只整理本次对话中明确出现的主诉；未问到则写未涉及",
    "history_of_present_illness": "只整理本次对话中明确出现的现病史；未问到则写未涉及",
    "menstrual_marital_reproductive_history": "分别写月经史、婚育史、性生活相关史；未问到的部分必须写未涉及",
    "past_history": "只整理本次对话中明确出现的既往史；未问到则写未涉及",
    "gynecological_history": "只整理本次对话中明确出现的妇科既往史、宫颈治疗史、阴道镜史；未问到则写未涉及",
    "screening_and_examination_history": "只整理本次对话中明确出现的HPV、TCT、阴道镜、HPV疫苗等信息；未问到则写未涉及",
    "student_initial_judgment": "复述学生初步判断",
    "student_colposcopy_decision": "复述学生阴道镜检查选择",
    "student_judgment_basis": "复述学生判断依据",
    "student_next_step_advice": "复述学生下一步建议",
    "system_reference_judgment": "基于本次对话给出教学参考判断，并说明信息不足之处"
  }},
  "dimension_scores": [
    {{
      "name": "问诊结构完整性",
      "score": 0,
      "max_score": 20,
      "comment": "评价是否覆盖主诉、现病史、月经婚育史、既往史等结构"
    }},
    {{
      "name": "妇科专科关键点覆盖",
      "score": 0,
      "max_score": 25,
      "comment": "评价是否覆盖异常出血、白带、HPV/TCT、阴道镜史等妇科关键点"
    }},
    {{
      "name": "临床思维与阴道镜指征判断",
      "score": 0,
      "max_score": 25,
      "comment": "评价学生判断是否合理，以及依据是否充分"
    }},
    {{
      "name": "问诊逻辑与问题质量",
      "score": 0,
      "max_score": 15,
      "comment": "评价问题顺序、追问质量、是否聚焦"
    }},
    {{
      "name": "医患沟通与人文关怀",
      "score": 0,
      "max_score": 10,
      "comment": "评价语气、隐私保护、解释敏感问题等"
    }},
    {{
      "name": "报告完整性与表达规范",
      "score": 0,
      "max_score": 5,
      "comment": "评价学生提交判断是否清楚、依据是否规范"
    }}
  ],
  "overall_feedback": "总体教师式评价，100到200字左右。必须明确指出本次问诊是否充分。",
  "strengths": ["优点1", "优点2"],
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
        temperature=0.1,
        timeout=90.0
    )

    if content.startswith("【模型调用失败】") or content.startswith("【模型调用异常】") or content.startswith("【系统提示】"):
        raise ValueError(content)

    report_data = _safe_json_loads(content)

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
    structured_report.setdefault("chief_complaint", "未涉及")
    structured_report.setdefault("history_of_present_illness", "未涉及")
    structured_report.setdefault("menstrual_marital_reproductive_history", "未涉及")
    structured_report.setdefault("past_history", "未涉及")
    structured_report.setdefault("gynecological_history", "未涉及")
    structured_report.setdefault("screening_and_examination_history", "未涉及")
    structured_report.setdefault("student_initial_judgment", student_submission.get("initial_judgment", ""))
    structured_report.setdefault("student_colposcopy_decision", student_submission.get("colposcopy_decision", ""))
    structured_report.setdefault("student_judgment_basis", student_submission.get("judgment_basis", ""))
    structured_report.setdefault("student_next_step_advice", student_submission.get("next_step_advice", ""))
    structured_report.setdefault("system_reference_judgment", "未生成")

    report_data = _apply_report_constraints(report_data, coverage)

    return report_data