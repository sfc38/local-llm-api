import base64

import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Local LLM API Playground", layout="wide")
st.title("Local LLM API Playground")


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


# Sidebar
with st.sidebar:
    st.header("Settings")

    if st.button("Check API Health"):
        try:
            r = httpx.get(f"{API_URL}/health", timeout=5)
            data = r.json()
            if r.status_code == 200:
                st.success(f"OK — model: {data.get('model')}")
            else:
                st.error(f"Error: {data}")
        except httpx.ConnectError:
            st.error("API not reachable")

    st.divider()
    model = st.text_input("Model", value="qwen2.5vl:3b")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    max_tokens = st.slider("Max tokens", 64, 2048, 512, 64)


def base_payload(**kwargs) -> dict:
    return {"model": model, "temperature": temperature, "max_tokens": max_tokens, **kwargs}


# Tabs
tab_chat, tab_gen, tab_img, tab_sum, tab_cls, tab_kw = st.tabs(
    ["Chat", "Generate", "Describe Image", "Summarize", "Classify", "Extract Keywords"]
)

with tab_chat:
    st.subheader("Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Clear conversation", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type a message...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        payload = base_payload(messages=st.session_state.chat_history)
        with st.chat_message("assistant"):
            output = st.empty()
            response_text = ""
            for token in stream_response("chat", payload):
                response_text += token
                output.markdown(response_text + "▌")
            output.markdown(response_text)

        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

with tab_gen:
    st.subheader("Generate")
    prompt = st.text_area("Prompt", height=150, key="gen_prompt")
    uploaded = st.file_uploader("Image (optional)", type=["jpg", "jpeg", "png"], key="gen_img")
    if st.button("Generate", key="gen_btn"):
        if not prompt:
            st.warning("Enter a prompt.")
        else:
            payload = base_payload(prompt=prompt)
            if uploaded:
                payload["images"] = [base64.b64encode(uploaded.read()).decode()]
            render_stream("generate", payload)

with tab_img:
    st.subheader("Describe Image")
    uploaded_img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="desc_img")
    desc_prompt = st.text_input("Prompt", value="Describe this image in detail.", key="desc_prompt")
    if st.button("Describe", key="desc_btn"):
        if not uploaded_img:
            st.warning("Upload an image first.")
        else:
            img_bytes = uploaded_img.read()
            st.image(img_bytes, caption="Input image", width=300)
            b64 = base64.b64encode(img_bytes).decode()
            render_stream("describe-image", base_payload(image=b64, prompt=desc_prompt))

with tab_sum:
    st.subheader("Summarize")
    text_input = st.text_area("Text to summarize", height=200, key="sum_text")
    if st.button("Summarize", key="sum_btn"):
        if not text_input:
            st.warning("Enter some text.")
        else:
            render_stream("summarize", base_payload(text=text_input))

with tab_cls:
    st.subheader("Classify")
    cls_text = st.text_area("Text to classify", height=150, key="cls_text")
    cats_input = st.text_input(
        "Categories (comma-separated)", value="finance, sports, technology, politics", key="cls_cats"
    )
    if st.button("Classify", key="cls_btn"):
        if not cls_text or not cats_input:
            st.warning("Enter text and categories.")
        else:
            categories = [c.strip() for c in cats_input.split(",") if c.strip()]
            render_stream("classify", base_payload(text=cls_text, categories=categories))

with tab_kw:
    st.subheader("Extract Keywords")
    kw_text = st.text_area("Text", height=150, key="kw_text")
    if st.button("Extract Keywords", key="kw_btn"):
        if not kw_text:
            st.warning("Enter some text.")
        else:
            render_stream("extract-keywords", base_payload(text=kw_text))
