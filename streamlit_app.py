import base64
import io

import httpx
import pandas as pd
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Local LLM API Playground", layout="wide")


# ── Helpers ──────────────────────────────────────────────────────────────────

def stream_response(endpoint: str, payload: dict):
    try:
        with httpx.Client(timeout=120) as client:
            with client.stream("POST", f"{API_URL}/{endpoint}", json=payload) as r:
                if r.status_code != 200:
                    r.read()
                    yield f"[Error {r.status_code}: {r.text}]"
                    return
                for line in r.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        yield line[6:]
    except httpx.ConnectError:
        yield "[Error: API server not reachable. Run: uvicorn app.main:app --reload]"
    except Exception as e:
        yield f"[Error: {e}]"


def render_stream(endpoint: str, payload: dict):
    output = st.empty()
    response_text = ""
    for token in stream_response(endpoint, payload):
        response_text += token
        output.markdown(response_text + "▌")
    output.markdown(response_text)


@st.cache_data(ttl=30)
def fetch_models() -> list[str]:
    try:
        r = httpx.get(f"{API_URL}/models", timeout=5)
        if r.status_code == 200:
            return r.json().get("models", [])
    except Exception:
        pass
    return ["qwen2.5vl:3b"]


def extract_pdf_text(file) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as e:
        return f"[Could not extract PDF text: {e}]"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("LLM Playground")

    page = st.radio(
        "Mode",
        ["Chat", "Generate", "Describe Image", "Summarize", "Classify", "Extract Keywords", "Reports"],
    )

    st.divider()

    if st.button("Check API Health"):
        try:
            r = httpx.get(f"{API_URL}/health", timeout=5)
            data = r.json()
            if r.status_code == 200:
                st.success(f"OK — {data.get('model')}")
            else:
                st.error(str(data))
        except httpx.ConnectError:
            st.error("API not reachable")

    st.divider()

    models = fetch_models()
    model = st.selectbox("Model", models) if models else st.text_input("Model", value="qwen2.5vl:3b")

    with st.expander("Parameters", expanded=True):

        temperature = st.slider(
            "🌡️ Temperature", 0.0, 2.0, 0.7, 0.05,
            help=(
                "Controls how random the model's word choices are.\n\n"
                "The model scores every possible next word. Temperature scales those scores "
                "before the model picks one.\n\n"
                "• 0.0 — always picks the highest-scoring word (deterministic)\n"
                "• 0.7 — good default, balanced creativity\n"
                "• 1.0+ — picks lower-ranked words more often, more surprising output\n"
                "• 2.0 — very random, may become incoherent\n\n"
                "Use low values for factual tasks (Q&A, code). "
                "Use higher values for creative writing."
            ),
        )
        if temperature < 0.3:
            st.caption("🎯 Very focused — nearly deterministic, best for facts and code")
        elif temperature < 0.8:
            st.caption("⚖️ Balanced — good default for most tasks")
        elif temperature < 1.3:
            st.caption("🎨 Creative — varied outputs, good for writing")
        else:
            st.caption("🌪️ Very random — outputs may become incoherent")

        max_tokens = st.slider(
            "📏 Max tokens", 64, 8192, 512, 64,
            help=(
                "The maximum number of tokens (roughly ¾ of a word each) the model will generate "
                "in a single response. The model may stop earlier if it finishes naturally.\n\n"
                "• 64–256 — short answers, classifications\n"
                "• 512 — good default\n"
                "• 1024–2048 — long explanations or summaries\n"
                "• 4096+ — very long documents (slow)\n\n"
                "Higher values do not make the model 'try harder' — they just allow longer output."
            ),
        )
        if max_tokens <= 256:
            st.caption("✂️ Short — responses will be cut off if they run long")
        elif max_tokens <= 1024:
            st.caption("📄 Standard — good for most responses")
        else:
            st.caption("📚 Long — allows detailed or document-length output")

        top_p = st.slider(
            "🎯 Top-p (nucleus sampling)", 0.0, 1.0, 0.9, 0.05,
            help=(
                "At each step the model ranks all possible next words by probability. "
                "Top-p cuts off the list once the cumulative probability reaches p, "
                "then samples only from that shorter list.\n\n"
                "• 0.1 — only the very top words are considered (conservative)\n"
                "• 0.9 — considers a wide range of plausible words (default)\n"
                "• 1.0 — considers all words (no cut-off)\n\n"
                "Works together with temperature. Lowering top-p is another way to "
                "make output more focused without changing temperature."
            ),
        )
        if top_p < 0.5:
            st.caption("🔒 Conservative — only the most probable words considered")
        elif top_p < 0.95:
            st.caption("✅ Standard nucleus sampling")
        else:
            st.caption("🌐 Open — almost all words are in the candidate pool")

        top_k = st.slider(
            "🔢 Top-k", 1, 100, 40, 1,
            help=(
                "At each step, only the top-k highest-probability words are considered "
                "(regardless of their actual probabilities). Then temperature and top-p "
                "are applied within that set.\n\n"
                "• 1 — always picks the single best word (greedy, like temperature 0)\n"
                "• 10–20 — focused, fewer surprises\n"
                "• 40 — good default\n"
                "• 100 — wide candidate pool\n\n"
                "Think of top-k as a hard limit, top-p as a soft probability limit. "
                "Both are applied, whichever is more restrictive wins."
            ),
        )
        if top_k <= 5:
            st.caption("🔒 Very restrictive — only top few words considered")
        elif top_k <= 20:
            st.caption("🎯 Focused token selection")
        elif top_k <= 60:
            st.caption("✅ Standard range")
        else:
            st.caption("🌐 Wide candidate pool")

        repeat_penalty = st.slider(
            "🔁 Repeat penalty", 0.5, 2.0, 1.1, 0.05,
            help=(
                "Discourages the model from repeating words or phrases it has already used "
                "in the response. Applied by reducing the score of tokens that appeared recently.\n\n"
                "• 0.5–0.9 — model can repeat freely (use for structured output like lists)\n"
                "• 1.0 — no penalty (neutral)\n"
                "• 1.1 — mild penalty (default, reduces redundancy)\n"
                "• 1.5+ — strong penalty, may produce awkward avoidance of common words\n"
                "• 2.0 — very aggressive, usually too strong\n\n"
                "If the model is looping or repeating itself, increase this. "
                "If output feels unnatural, decrease it."
            ),
        )
        if repeat_penalty < 1.0:
            st.caption("♻️ Below 1 — repetition is encouraged (unusual)")
        elif repeat_penalty < 1.2:
            st.caption("✅ Mild penalty — standard, reduces redundancy naturally")
        elif repeat_penalty < 1.6:
            st.caption("🚫 Strong penalty — good for fixing looping models")
        else:
            st.caption("⚠️ Very aggressive — may produce unnatural word choices")

        num_ctx = st.slider(
            "📦 Context window (num_ctx)", 512, 32768, 4096, 512,
            help=(
                "How many tokens of conversation history the model can 'see' at once. "
                "This includes both your messages and the model's replies.\n\n"
                "• 512–1024 — tiny, model forgets early parts of a long chat quickly\n"
                "• 2048 — Ollama's default (often too small for real conversations)\n"
                "• 4096 — good starting point for chat\n"
                "• 8192–16384 — long documents, multi-page chats\n"
                "• 32768 — maximum for Qwen2.5-VL 3B\n\n"
                "Larger context uses more RAM and makes each response slower. "
                "For one-off prompts, 2048 is fine. For ongoing conversations, use at least 4096."
            ),
        )
        if num_ctx <= 1024:
            st.caption("⚡ Tiny context — fast but forgets quickly in long chats")
        elif num_ctx <= 2048:
            st.caption("📌 Ollama default — fine for short interactions")
        elif num_ctx <= 8192:
            st.caption("💬 Good for conversations and medium documents")
        else:
            st.caption("📚 Large context — slow but can handle long documents")

        seed_raw = st.number_input(
            "🎲 Seed (-1 = random)", min_value=-1, value=-1, step=1,
            help=(
                "Sets the random seed for the model's sampling process.\n\n"
                "• -1 (default) — different output every run\n"
                "• Any fixed number (e.g. 42) — same output every time for the same input\n\n"
                "Useful for: debugging prompts (compare outputs fairly), "
                "demonstrations where you want consistent results, "
                "or A/B testing parameter changes while keeping everything else constant.\n\n"
                "Note: Only works reliably when temperature > 0. At temperature 0 "
                "the model is already deterministic regardless of seed."
            ),
        )
        seed = None if seed_raw == -1 else int(seed_raw)
        if seed is None:
            st.caption("🎲 Random — different output each run")
        else:
            st.caption(f"📌 Fixed seed {seed} — reproducible output")


def base_payload(**kwargs) -> dict:
    p = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "num_ctx": num_ctx,
    }
    if seed is not None:
        p["seed"] = seed
    p.update(kwargs)
    return p


# ── Chat ──────────────────────────────────────────────────────────────────────

if page == "Chat":
    st.title("Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "file_key" not in st.session_state:
        st.session_state.file_key = 0
    if "pending_attach" not in st.session_state:
        st.session_state.pending_attach = None

    submitted = False
    user_input = ""

    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.pending_attach = None
        st.rerun()

    # Render conversation history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg.get("_images"):
                for img_b64 in msg["_images"]:
                    st.image(base64.b64decode(img_b64), width=250)
            if msg.get("_pdf_name"):
                st.caption(f"📄 {msg['_pdf_name']}")
            st.markdown(msg.get("_user_text", msg["content"]))

    # ── Bottom bar: 📎 popover + text input, pinned to viewport bottom ─────────
    with st.bottom:
        if st.session_state.pending_attach:
            attach_info = st.session_state.pending_attach
            icon = "🖼️" if attach_info["type"] == "image" else "📄"
            a_col, x_col = st.columns([11, 1])
            with a_col:
                st.info(f"{icon} **{attach_info['name']}** will be attached to your next message.")
            with x_col:
                if st.button("✕", key="rm_attach", help="Remove attachment"):
                    st.session_state.pending_attach = None
                    st.rerun()

        pb_col, inp_col = st.columns([1, 11])
        with pb_col:
            with st.popover("📎", use_container_width=True):
                st.caption("Attach an image or PDF.")
                upl = st.file_uploader(
                    "File",
                    type=["jpg", "jpeg", "png", "pdf"],
                    key=f"chat_file_{st.session_state.file_key}",
                    label_visibility="collapsed",
                )
                if upl is not None:
                    raw = upl.read()
                    if upl.type == "application/pdf":
                        st.session_state.pending_attach = {"type": "pdf", "name": upl.name, "data": raw}
                    else:
                        st.session_state.pending_attach = {
                            "type": "image",
                            "name": upl.name,
                            "data": base64.b64encode(raw).decode(),
                        }
                    st.session_state.file_key += 1
                    st.rerun()
        with inp_col:
            with st.form("chat_form", clear_on_submit=True, border=False):
                f_text, f_send = st.columns([10, 1])
                with f_text:
                    user_input = st.text_input(
                        "message",
                        placeholder="Type a message…",
                        label_visibility="collapsed",
                    )
                with f_send:
                    submitted = st.form_submit_button("➤", use_container_width=True)

    if submitted and user_input:
        attach = st.session_state.pending_attach
        images = None
        pdf_name = None
        full_content = user_input

        if attach:
            if attach["type"] == "pdf":
                import io
                pdf_text = extract_pdf_text(io.BytesIO(attach["data"]))
                full_content = f"{user_input}\n\n[Attached PDF: {attach['name']}]\n{pdf_text}"
                pdf_name = attach["name"]
            else:
                images = [attach["data"]]

        msg = {"role": "user", "content": full_content, "_user_text": user_input}
        if images:
            msg["images"] = images
            msg["_images"] = images
        if pdf_name:
            msg["_pdf_name"] = pdf_name

        st.session_state.chat_history.append(msg)
        st.session_state.pending_attach = None

        with st.chat_message("user"):
            if images:
                st.image(base64.b64decode(images[0]), width=250)
            if pdf_name:
                st.caption(f"📄 {pdf_name}")
            st.markdown(user_input)

        api_messages = [
            {k: v for k, v in m.items() if not k.startswith("_")}
            for m in st.session_state.chat_history
        ]
        payload = base_payload(messages=api_messages)

        with st.chat_message("assistant"):
            output = st.empty()
            response_text = ""
            for token in stream_response("chat", payload):
                response_text += token
                output.markdown(response_text + "▌")
            output.markdown(response_text)

        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

# ── Generate ──────────────────────────────────────────────────────────────────

elif page == "Generate":
    st.title("Generate")
    prompt = st.text_area("Prompt", height=150)
    uploaded = st.file_uploader("Image (optional)", type=["jpg", "jpeg", "png"])
    if st.button("Generate"):
        if not prompt:
            st.warning("Enter a prompt.")
        else:
            payload = base_payload(prompt=prompt)
            if uploaded:
                payload["images"] = [base64.b64encode(uploaded.read()).decode()]
            render_stream("generate", payload)

# ── Describe Image ────────────────────────────────────────────────────────────

elif page == "Describe Image":
    st.title("Describe Image")
    uploaded_img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    desc_prompt = st.text_input("Prompt", value="Describe this image in detail.")
    if st.button("Describe"):
        if not uploaded_img:
            st.warning("Upload an image first.")
        else:
            img_bytes = uploaded_img.read()
            st.image(img_bytes, caption="Input image", width=300)
            b64 = base64.b64encode(img_bytes).decode()
            render_stream("describe-image", base_payload(image=b64, prompt=desc_prompt))

# ── Summarize ─────────────────────────────────────────────────────────────────

elif page == "Summarize":
    st.title("Summarize")
    text_input = st.text_area("Text to summarize", height=200)
    if st.button("Summarize"):
        if not text_input:
            st.warning("Enter some text.")
        else:
            render_stream("summarize", base_payload(text=text_input))

# ── Classify ──────────────────────────────────────────────────────────────────

elif page == "Classify":
    st.title("Classify")
    cls_text = st.text_area("Text to classify", height=150)
    cats_input = st.text_input("Categories (comma-separated)", value="finance, sports, technology, politics")
    if st.button("Classify"):
        if not cls_text or not cats_input:
            st.warning("Enter text and categories.")
        else:
            categories = [c.strip() for c in cats_input.split(",") if c.strip()]
            render_stream("classify", base_payload(text=cls_text, categories=categories))

# ── Extract Keywords ──────────────────────────────────────────────────────────

elif page == "Extract Keywords":
    st.title("Extract Keywords")
    kw_text = st.text_area("Text", height=150)
    if st.button("Extract Keywords"):
        if not kw_text:
            st.warning("Enter some text.")
        else:
            render_stream("extract-keywords", base_payload(text=kw_text))

# ── Reports ───────────────────────────────────────────────────────────────────

elif page == "Reports":
    st.title("Reports")

    try:
        r = httpx.get(f"{API_URL}/reports", timeout=5)
        data = r.json().get("requests", [])
    except Exception as e:
        st.error(f"Could not load reports: {e}")
        data = []

    if not data:
        st.info("No requests logged yet. Use the other pages to generate data.")
    else:
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Requests", len(df))
        col2.metric("Avg Response Time", f"{df['response_time_ms'].mean():.0f} ms")
        col3.metric("Endpoints Used", df["endpoint"].nunique())
        col4.metric("Success Rate", f"{(df['status_code'] == 200).mean() * 100:.0f}%")

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Avg Response Time by Endpoint")
            avg_time = df.groupby("endpoint")["response_time_ms"].mean().sort_values()
            st.bar_chart(avg_time)

        with col_b:
            st.subheader("Request Count by Endpoint")
            counts = df["endpoint"].value_counts()
            st.bar_chart(counts)

        st.divider()
        st.subheader("All Requests")
        st.dataframe(
            df[["timestamp", "endpoint", "model", "prompt_preview", "response_time_ms", "status_code"]]
            .rename(columns={"response_time_ms": "time_ms", "prompt_preview": "prompt"}),
            use_container_width=True,
        )

        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "requests.csv", "text/csv")
