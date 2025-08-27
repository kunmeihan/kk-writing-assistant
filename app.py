# app.py — KK 写作助手（接受/保留后折叠为“取消”；黄色高亮；单一预览；不改输入原文）
import os, json, html, difflib, uuid, re
from typing import List, Dict, Any
import streamlit as st
from dotenv import load_dotenv

# ========= 环境与 OpenAI（可选）=========
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
try:
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY) if API_KEY else None
except Exception:
    client = None

st.set_page_config(page_title="KK 写作助手", layout="wide")
st.title("KK 写作助手")

# --- 侧栏设置 ---
dialect = st.sidebar.selectbox("拼写风格", ["American", "British"], index=0)
style = st.sidebar.selectbox("写作风格", ["Academic", "Formal", "Neutral", "Friendly", "Concise"], index=2)
audience = st.sidebar.selectbox("读者类型", ["General", "Expert", "Educator", "Business"], index=0)
DEBUG = st.sidebar.checkbox("显示调试信息", value=False)

def debug_show(title, obj):
    if DEBUG:
        st.sidebar.subheader(title)
        try:
            st.sidebar.json(obj)
        except Exception:
            st.sidebar.write(obj)

# ============ 高亮（仅黄色显示新文本） ============
def highlight_yellow(original: str, modified: str) -> str:
    """仅对新增/替换的新文本做黄色高亮；删除内容不显示。"""
    parts = []
    s = difflib.SequenceMatcher(None, original, modified)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            parts.append(html.escape(original[i1:i2]))
        elif tag in ('replace', 'insert'):
            parts.append(f"<mark>{html.escape(modified[j1:j2])}</mark>")
    return "".join(parts)

# ============ 小工具 ============
def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

def _is_word_char(ch: str) -> bool:
    return ch.isalpha() or ch.isdigit() or ch == "_"

def _boundary_score(text: str, start: int, end: int) -> int:
    left_ok = (start == 0) or (not _is_word_char(text[start - 1]))
    right_ok = (end >= len(text)) or (not _is_word_char(text[end]))
    return int(left_ok) + int(right_ok)

def _find_all(text: str, needle: str):
    idxs, i = [], 0
    if not needle:
        return idxs
    while True:
        i = text.find(needle, i)
        if i == -1: break
        idxs.append(i); i += 1
    return idxs

def realign_span(text: str, s: dict, window: int = 120):
    """
    若 s['before'] 与 text[start:end] 不一致：附近/全局精确匹配；不行再 trim 匹配。
    多处命中 → 词边界优先，其次距离 start 近。
    返回 (start, end, used_trim) 或 (None, None, False)。
    """
    before = s.get("before") or ""
    start = safe_int(s.get("start"), -1)
    end   = safe_int(s.get("end"), -1)

    if before == "":
        if 0 <= start <= len(text): return start, start, False
        return None, None, False

    def pick_best(cands, approx, needle):
        if not cands: return None
        scored = []
        for stx in cands:
            edx = stx + len(needle)
            score = _boundary_score(text, stx, edx)
            dist = abs(stx - max(0, approx))
            scored.append((score, -dist, stx))
        scored.sort(reverse=True)
        return scored[0][2]

    # 原位
    if 0 <= start <= end <= len(text) and text[start:end] == before:
        return start, end, False

    approx = start if 0 <= start <= len(text) else 0
    lo, hi = max(0, approx-window), min(len(text), approx+window)

    near = _find_all(text[lo:hi], before)
    if near:
        best = pick_best([lo+i for i in near], approx, before)
        if best is not None: return best, best+len(before), False

    allx = _find_all(text, before)
    if allx:
        best = pick_best(allx, approx, before)
        if best is not None: return best, best+len(before), False

    trimmed = before.strip()
    if trimmed and trimmed != before:
        near = _find_all(text[lo:hi], trimmed)
        if near:
            best = pick_best([lo+i for i in near], approx, trimmed)
            if best is not None: return best, best+len(trimmed), True
        allx = _find_all(text, trimmed)
        if allx:
            best = pick_best(allx, approx, trimmed)
            if best is not None: return best, best+len(trimmed), True

    return None, None, False

# ============ 全局状态（不改输入原文） ============
if "source_text" not in st.session_state:           # 输入框文本（仅用户改）
    st.session_state.source_text = ""
if "analysis_source_text" not in st.session_state:  # 生成建议时的快照
    st.session_state.analysis_source_text = ""
if "revision_text" not in st.session_state:         # 修订稿（=按“已接受集合”从快照重算）
    st.session_state.revision_text = ""
if "suggestions" not in st.session_state:
    st.session_state.suggestions: List[Dict[str, Any]] = []
if "accepted_ids" not in st.session_state:
    st.session_state.accepted_ids = set()
if "kept_ids" not in st.session_state:              # 保留原文集合
    st.session_state.kept_ids = set()

# ============ 稳定 UI：保持展开 ============
def keep_open(sid: str):
    st.session_state[f"exp_open_{sid}"] = True

# ============ 空格/标点与去重（通用） ============
_PUNCT_NEED_SPACE_AFTER = {",", ";", ":"}

def apply_one_change(out: str, rs: int, re_: int, before: str, rep: str) -> str:
    """
    在 out 上应用一次替换，包含：
    - 去重保护（避免把切片再写一遍）
    - 仅在“插入逗号/分号/冒号”时，必要才补一个空格
    - 避免左侧双空格
    不做“词边界收缩”，以免误删后词。
    """
    slice_text = out[rs:re_]

    # 去重保护
    if slice_text and rep.startswith(slice_text):
        rep = rep[len(slice_text):]
    elif rs == re_:
        # 纯插入，若 rep 重复左侧单词开头，去掉重复部分
        j = rs - 1
        while j >= 0 and out[j].isalnum(): j -= 1
        prev_token = out[j+1:rs]
        if prev_token and rep.startswith(prev_token):
            rep = rep[len(prev_token):]

    # 仅对 , ; : 的插入/替换尾部做空格补齐（其他不改用户空格）
    next_char = out[re_] if re_ < len(out) else ""
    if rep:
        last = rep[-1]
        if last in _PUNCT_NEED_SPACE_AFTER:
            if (next_char and next_char not in " \t\n\r,.;:!?)”’]}"):
                rep = rep + " "

    # 避免左侧双空格
    if rep.startswith(" ") and rs > 0 and out[rs-1] == " ":
        rep = rep.lstrip()

    return out[:rs] + rep + out[re_:]

# ============ 逐条实时对齐并应用（用于重算修订稿） ============
def apply_suggestions_preview(base: str, suggestions: List[Dict[str, Any]], selected_ids: set) -> str:
    """
    从右到左，对“当前 out 文本”逐条 realign 后再应用。
    这里的 selected_ids 即为「已接受集合」。
    """
    chosen = [s for s in suggestions if s.get("id") in selected_ids]
    chosen.sort(key=lambda s: safe_int(s.get("start"), -1), reverse=True)

    out = base
    for s in chosen:
        rs, re_, _ = realign_span(out, s)
        if rs is None:
            continue
        before = s.get("before", "")
        rep    = s.get("replacement", "")
        out = apply_one_change(out, rs, re_, before, rep)
    return out

# ============ 可逆：基于“已接受集合”重算修订稿 ============
def recompute_revision_from_accepts():
    """
    用分析时的输入快照作为基线，按已接受的 ID 集合重新生成修订稿。
    （可逆、稳定，不串位）
    """
    base = st.session_state.analysis_source_text or st.session_state.source_text
    st.session_state.revision_text = apply_suggestions_preview(
        base,
        st.session_state.suggestions,
        st.session_state.accepted_ids
    )

def set_status(sid: str, status: str):
    """
    三态切换：
      - 'accept'  : 加入 accepted_ids，移出 kept_ids
      - 'keep'    : 加入 kept_ids，移出 accepted_ids
      - 'neutral' : 两个集合都移除（即“取消”）
    然后基于“已接受集合”重算修订稿，并同步建议状态。
    """
    if status == "accept":
        st.session_state.kept_ids.discard(sid)
        st.session_state.accepted_ids.add(sid)
    elif status == "keep":
        st.session_state.accepted_ids.discard(sid)
        st.session_state.kept_ids.add(sid)
    elif status == "neutral":
        st.session_state.accepted_ids.discard(sid)
        st.session_state.kept_ids.discard(sid)

    recompute_revision_from_accepts()

    acc_set = st.session_state.accepted_ids
    kept_set = st.session_state.kept_ids
    for s in st.session_state.suggestions:
        sid0 = s.get("id")
        s["accepted"] = (sid0 in acc_set)
        s["kept"] = (sid0 in kept_set)

def set_status_and_keep_open(sid: str, status: str):
    st.session_state[f"exp_open_{sid}"] = True
    set_status(sid, status)

# ============ 本地兜底（英文常见问题） ============
def analyze_text_locally(text: str) -> dict:
    sugs = []
    def add(id_pref, start, end, before, after, category, explanation, suggestion, severity="low"):
        sugs.append({
            "id": f"{id_pref}_{uuid.uuid4().hex[:8]}",
            "category": category,
            "severity": severity,
            "issue_excerpt": text[max(0, start-12):min(len(text), max(end, start)+12)],
            "explanation": explanation,
            "suggestion": suggestion,
            "before": before,
            "after": after,
            "start": start,
            "end": end,
            "replacement": after
        })

    # 标点空格/重复标点/所有格/a-an/双空格
    for m in re.finditer(r"\s+([,.;:!?])", text):
        add("space_before_punct", m.start(), m.end(), text[m.start():m.end()], m.group(1),
            "Punctuation", "Unnecessary space before punctuation.", "Remove the space.")
    for m in re.finditer(r"([,.;:!?])(?!\s|$)", text):
        j = m.end()
        if j < len(text) and text[j] not in [' ', '"', "'", ')', ']', '}', '”', '’']:
            add("space_after_punct", j, j, "", " ",
                "Punctuation", "Missing space after punctuation.", "Insert a space.")
    for m in re.finditer(r"([!?.,])\1+", text):
        add("dup_punct", m.start(), m.end(), m.group(0), m.group(1),
            "Punctuation", "Repeated punctuation is informal.", "Use a single punctuation mark.")
    for m in re.finditer(r"\s+'s\b", text):
        add("possessive_space", m.start(), m.end(), m.group(0), "'s",
            "Grammar", "No space before possessive 's.", "Remove the space.")
    for m in re.finditer(r"\b([Aa])\s+([aeiouAEIOU])", text):
        seg = text[m.start():m.end()]
        after = f"an {m.group(2)}"
        add("article_an", m.start(), m.end(), seg, after,
            "Grammar", "Use 'an' before vowel sounds.", "Change 'a' to 'an'.")
    for m in re.finditer(r" {2,}", text):
        add("double_space", m.start(), m.end(), m.group(0), " ",
            "Formatting", "Consecutive spaces reduce readability.", "Use a single space.")

    improved = apply_suggestions_preview(text, sugs, {s["id"] for s in sugs if s["replacement"]})
    return {"improved_text": improved or text, "suggestions": sugs}

# ============ OpenAI（结构化英文建议；JSON 模式） ============
def analyze_text_with_openai(text: str, dialect: str, style: str, audience: str) -> dict:
    prompt = f"""
You are an English writing editor. Output ONLY a json object (no prose).
Target dialect: {dialect} English. Style: {style}. Audience: {audience}.

Return:
{{
  "improved_text": "<text after applying all replacements>",
  "suggestions": [
    {{
      "id": "s-unique-id",
      "category": "Grammar|Punctuation|Clarity|Concision|Style|Tone|Word choice|Agreement|Spelling",
      "severity": "low|medium|high",
      "issue_excerpt": "short problematic span",
      "explanation": "why (1–2 sentences)",
      "suggestion": "how to fix (1–2 sentences)",
      "before": "exactly text[start:end]",
      "after": "proposed span",
      "start": <int, 0-based, inclusive>,
      "end": <int, exclusive>,
      "replacement": "same as 'after'"
    }}
  ]
}}
Rules:
- Indices must match the ORIGINAL text exactly.
- Ensure 'before' == text[start:end]; 'replacement' == 'after'.
- 'improved_text' = ORIGINAL with all replacements applied.
- Even good writing: provide 2–3 small improvements (punctuation, concision, word choice).
ORIGINAL:
{text}
"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a rigorous English writing editor. Output valid json only."},
            {"role": "user", "content": prompt}
        ]
    )
    raw = resp.choices[0].message.content
    debug_show("LLM 原始返回", raw)
    data = json.loads(raw)

    cleaned = []
    for s in data.get("suggestions", []) if isinstance(data, dict) else []:
        s["id"] = s.get("id") or f"s_{uuid.uuid4().hex[:8]}"
        s["start"] = safe_int(s.get("start"), 0)
        s["end"]   = safe_int(s.get("end"), s["start"])
        s["replacement"] = s.get("replacement") or s.get("after", "")
        if 0 <= s["start"] <= s["end"] <= len(text) and text[s["start"]:s["end"]] == (s.get("before") or text[s["start"]:s["end"]]):
            cleaned.append(s)
        else:
            rs, re_, used_trim = realign_span(text, s)
            if rs is not None:
                s["start"], s["end"] = rs, re_
                cleaned.append(s)
            else:
                debug_show("丢弃：无法对齐的建议", s)

    data["suggestions"] = cleaned
    if not data.get("improved_text"):
        data["improved_text"] = apply_suggestions_preview(text, cleaned, {x["id"] for x in cleaned})
    debug_show("LLM 解析后", data)
    return data

def analyze_text(text: str, dialect="American", style="Neutral", audience="General") -> dict:
    if client is not None:
        try:
            return analyze_text_with_openai(text, dialect, style, audience)
        except Exception as e:
            debug_show("OpenAI 出错 → 使用本地规则", str(e))
    else:
        debug_show("提示", "未检测到 OPENAI_API_KEY，使用本地兜底规则。")
    return analyze_text_locally(text)

def optimize_text(text: str, dialect="American", style="Neutral", audience="General") -> str:
    if client is None:
        return text
    try:
        prompt = f"""Please rewrite the text to improve clarity, coherence, and concision.
Follow {dialect} English conventions. Style: {style}. Audience: {audience}.
Return only the improved text.

Original:
{text}
"""
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are an English writing editor."},
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        debug_show("整体改写出错", str(e))
        return text

# ============ UI ============
# 输入框标题保持英文，且**不会被程序改写**
source_text = st.text_area("Paste or type your text here:", height=260, key="source_text")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    run_btn = st.button("分析并给出建议", type="primary", use_container_width=True)
with col2:
    preview_btn = st.button("预览整体改写", use_container_width=True)
with col3:
    st.caption("提示：输入原文后点击“分析”；在下方“预览”中查看效果，输入框不会被改动。")

# 生成建议（建立修订稿基线，并清空集合）
if run_btn:
    if not source_text.strip():
        st.warning("请输入文本。")
    else:
        with st.spinner("分析中…"):
            data = analyze_text(source_text, dialect, style, audience)
            st.session_state.suggestions = data.get("suggestions", [])
            st.session_state.analysis_source_text = source_text  # 记录输入快照
            st.session_state.accepted_ids = set()
            st.session_state.kept_ids = set()
            for s in st.session_state.suggestions:
                s.pop("accepted", None)
                s.pop("kept", None)
            st.session_state.revision_text = source_text
            st.success(f"已生成 {len(st.session_state.suggestions)} 条建议。")

# 整体改写（不改输入原文）
if preview_btn:
    if not source_text.strip():
        st.warning("请输入文本。")
    else:
        with st.spinner("改写中…"):
            improved = optimize_text(source_text, dialect, style, audience)
            st.subheader("整体改写预览（不落地）")
            st.text_area("改写后的文本：", value=improved, height=160)

# 建议 + 单一“预览”区
if st.session_state.suggestions:
    left, right = st.columns([1, 1], gap="large")

    # 若用户改动了输入框，禁止基于旧建议继续操作
    is_source_dirty = (st.session_state.analysis_source_text != st.session_state.source_text)

    with left:
        st.subheader("结构化建议")
        if is_source_dirty:
            st.warning("输入框内容已与分析时不同。请先重新分析后再操作。")

        for s in st.session_state.suggestions:
            sid = s.get("id"); cat = s.get("category","?")
            sev = s.get("severity","?"); issue = s.get("issue_excerpt","")[:60]
            is_accepted = (sid in st.session_state.accepted_ids)
            is_kept     = (sid in st.session_state.kept_ids)

            expanded_state = st.session_state.get(f"exp_open_{sid}", True)
            with st.expander(f"[{sev}] {cat} — {issue}", expanded=expanded_state):
                st.write("**原因（Why）**：", s.get("explanation",""))
                st.write("**方法（How）**：", s.get("suggestion",""))
                st.code(f"{s.get('before','')}  =>  {s.get('after','')}", language="text")

                # UI 逻辑：中立态 -> 显示两个按钮；接受或保留后 -> 折叠成“取消”
                if is_accepted or is_kept:
                    status_tip = "✅ 已接受" if is_accepted else "🧷 已保留"
                    st.caption(status_tip)
                    st.button(
                        "取消",
                        key=f"cancel_{sid}",
                        on_click=set_status_and_keep_open,
                        args=(sid, "neutral"),
                        use_container_width=True,
                        disabled=is_source_dirty
                    )
                else:
                    cols = st.columns([1, 1])
                    with cols[0]:
                        st.button(
                            "接受修改",
                            key=f"acc_{sid}",
                            on_click=set_status_and_keep_open,
                            args=(sid, "accept"),
                            use_container_width=True,
                            disabled=is_source_dirty
                        )
                    with cols[1]:
                        st.button(
                            "保留原文",
                            key=f"keep_{sid}",
                            on_click=set_status_and_keep_open,
                            args=(sid, "keep"),
                            use_container_width=True,
                            disabled=is_source_dirty
                        )

    with right:
        st.subheader("修订后预览")
        # 预览内容 = 基于“已接受集合”从快照重算的修订稿
        recompute_revision_from_accepts()
        preview_text = st.session_state.revision_text
        st.text_area("预览文本：", value=preview_text, height=220)

        # 仅黄色高亮：展示“原始快照 vs 预览”的差异（不显示删除）
        st.subheader("差异高亮（仅黄色显示改动后的内容）")
        base = st.session_state.analysis_source_text or st.session_state.source_text
        css = "<style> mark { background: #fff3a6; padding: 0 2px; } .diffbox{padding:10px;border:1px solid #ddd;border-radius:8px;line-height:1.7;} </style>"
        diff_html = highlight_yellow(base, preview_text)
        st.markdown(css + f"<div class='diffbox'>{diff_html}</div>", unsafe_allow_html=True)
