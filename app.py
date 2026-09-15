```python
import os
import io
import json
import ast
import math
import operator
from datetime import datetime

import streamlit as st
from groq import Groq

# Optional RAG imports
try:
    import faiss
    import numpy as np
    from pypdf import PdfReader
    from sentence_transformers import SentenceTransformer

    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Nova Voice AI Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(88, 28, 135, 0.18), transparent 25%),
            radial-gradient(circle at 90% 20%, rgba(37, 99, 235, 0.15), transparent 25%),
            #080b12;
        color: #f8fafc;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .status-card {
        padding: 15px;
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.15);
        margin-bottom: 12px;
    }

    .memory-card {
        padding: 12px;
        border-radius: 12px;
        background: rgba(30, 41, 59, 0.65);
        border-left: 3px solid #8b5cf6;
        margin-bottom: 8px;
    }

    .tool-card {
        padding: 10px;
        border-radius: 10px;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(96, 165, 250, 0.2);
    }

    div[data-testid="stChatMessage"] {
        border-radius: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

CHAT_MODEL = "openai/gpt-oss-20b"
STT_MODEL = "whisper-large-v3-turbo"

SYSTEM_PROMPT = """
You are Nova, a professional real-time AI voice assistant.

Your personality:
- Helpful
- Friendly
- Concise but useful
- Natural in conversation
- Professional

You can:
1. Answer normal questions.
2. Remember information the user explicitly shares.
3. Use tools when useful.
4. Answer questions about uploaded documents.
5. Perform simple calculations.
6. Tell the current date/time when needed.

Memory rules:
- If the user tells you a stable personal preference, name, goal, skill,
  project, or other useful fact, remember it.
- Do not invent personal information.
- Do not claim to remember something that is not present in the supplied memory.
- Respect user requests to forget information.

For uploaded documents:
- Prefer the supplied document context when answering document questions.
- If the answer is not present, say that it was not found in the uploaded documents.

Keep responses reasonably concise because they may be spoken aloud.
"""


# ============================================================
# SESSION STATE
# ============================================================

def initialize_state():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = []

    if "rag_chunks" not in st.session_state:
        st.session_state.rag_chunks = []

    if "rag_index" not in st.session_state:
        st.session_state.rag_index = None

    if "rag_files" not in st.session_state:
        st.session_state.rag_files = []

    if "embedder" not in st.session_state:
        st.session_state.embedder = None

    if "last_audio_id" not in st.session_state:
        st.session_state.last_audio_id = None


initialize_state()


# ============================================================
# API KEY
# ============================================================

def get_api_key():

    # Streamlit Secrets first
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    # Environment variable
    env_key = os.getenv("GROQ_API_KEY")

    if env_key:
        return env_key

    return None


# ============================================================
# GROQ CLIENT
# ============================================================

def create_client(api_key):

    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Could not initialize Groq: {e}")
        return None


# ============================================================
# SAFE CALCULATOR TOOL
# ============================================================

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calculate(expression):

    expression = expression.strip()

    if len(expression) > 200:
        raise ValueError("Expression is too long.")

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):

        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Invalid constant.")

        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)

            operation = ALLOWED_OPERATORS.get(type(node.op))

            if not operation:
                raise ValueError("Operator not allowed.")

            return operation(left, right)

        if isinstance(node, ast.UnaryOp):
            operation = ALLOWED_OPERATORS.get(type(node.op))

            if not operation:
                raise ValueError("Operator not allowed.")

            return operation(evaluate(node.operand))

        if isinstance(node, ast.Call):
            raise ValueError("Functions are not allowed.")

        raise ValueError("Invalid expression.")

    result = evaluate(tree)

    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("Invalid mathematical result.")

    return result


# ============================================================
# TOOL DEFINITIONS
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate a mathematical expression safely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression such as 25*4+10",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_datetime",
            "description": "Get the current local date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


def execute_tool(name, arguments):

    try:

        if name == "calculator":

            expression = arguments.get("expression", "")
            result = safe_calculate(expression)

            return str(result)

        if name == "current_datetime":

            return datetime.now().strftime(
                "%A, %d %B %Y at %I:%M %p"
            )

        return "Unknown tool."

    except Exception as e:

        return f"Tool error: {e}"


# ============================================================
# MEMORY
# ============================================================

def add_memory(text):

    text = text.strip()

    if not text:
        return

    if text not in st.session_state.memory:

        st.session_state.memory.append(text)

        # Keep memory lightweight
        if len(st.session_state.memory) > 30:
            st.session_state.memory = st.session_state.memory[-30:]


def detect_memory(user_text):

    """
    Lightweight long-term memory extraction.

    This intentionally avoids another LLM request so the application
    remains cheap and simple.
    """

    text = user_text.strip()

    lower = text.lower()

    memory_triggers = [
        "my name is ",
        "i am ",
        "i'm ",
        "i work as ",
        "i work at ",
        "i study ",
        "i am studying ",
        "i study at ",
        "i live in ",
        "i'm from ",
        "i like ",
        "i love ",
        "my favorite ",
        "my goal is ",
        "i want to become ",
        "i am learning ",
        "i'm learning ",
        "my project is ",
    ]

    for trigger in memory_triggers:

        if trigger in lower:

            # Avoid saving extremely long statements.
            if len(text) <= 250:
                add_memory(text)

            break


def build_memory_context():

    if not st.session_state.memory:
        return "No long-term user memory is currently available."

    return "\n".join(
        f"- {item}"
        for item in st.session_state.memory
    )


# ============================================================
# RAG
# ============================================================

@st.cache_resource
def load_embedder():

    if not RAG_AVAILABLE:
        return None

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


def split_text(text, chunk_size=900, overlap=150):

    text = text.replace("\x00", " ")

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def process_pdfs(files):

    if not RAG_AVAILABLE:
        st.error(
            "RAG dependencies are unavailable. "
            "Please install requirements.txt."
        )
        return

    all_chunks = []
    file_names = []

    for uploaded_file in files:

        try:

            pdf_bytes = uploaded_file.getvalue()

            reader = PdfReader(io.BytesIO(pdf_bytes))

            full_text = ""

            for page in reader.pages:

                page_text = page.extract_text() or ""

                full_text += "\n" + page_text

            chunks = split_text(full_text)

            for chunk in chunks:

                all_chunks.append(
                    {
                        "text": chunk,
                        "source": uploaded_file.name,
                    }
                )

            file_names.append(uploaded_file.name)

        except Exception as e:

            st.warning(
                f"Could not process {uploaded_file.name}: {e}"
            )

    if not all_chunks:
        st.warning("No readable text was found in the uploaded PDFs.")
        return

    try:

        embedder = load_embedder()

        texts = [
            item["text"]
            for item in all_chunks
        ]

        embeddings = embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        index.add(embeddings)

        st.session_state.rag_chunks = all_chunks
        st.session_state.rag_index = index
        st.session_state.embedder = embedder
        st.session_state.rag_files = file_names

    except Exception as e:

        st.error(f"RAG indexing failed: {e}")


def search_documents(query, top_k=4):

    if (
        st.session_state.rag_index is None
        or not st.session_state.rag_chunks
    ):
        return []

    try:

        embedder = st.session_state.embedder

        query_embedding = embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = st.session_state.rag_index.search(
            query_embedding,
            min(
                top_k,
                len(st.session_state.rag_chunks)
            ),
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index >= 0:

                item = st.session_state.rag_chunks[index]

                results.append(
                    {
                        "text": item["text"],
                        "source": item["source"],
                        "score": float(score),
                    }
                )

        return results

    except Exception:
        return []


def build_rag_context(user_text):

    results = search_documents(user_text)

    if not results:
        return ""

    context = "\n\n".join(
        f"[Source: {item['source']}]\n{item['text']}"
        for item in results
    )

    return context


# ============================================================
# SPEECH TO TEXT
# ============================================================

def transcribe_audio(client, audio_file):

    try:

        audio_bytes = audio_file.getvalue()

        transcription = client.audio.transcriptions.create(
            file=(
                "recording.wav",
                audio_bytes,
                "audio/wav",
            ),
            model=STT_MODEL,
            response_format="text",
        )

        return transcription.strip()

    except Exception as e:

        st.error(
            f"Speech recognition failed: {e}"
        )

        return ""


# ============================================================
# AI RESPONSE
# ============================================================

def generate_response(client, user_text):

    memory_context = build_memory_context()

    rag_context = build_rag_context(user_text)

    system_message = SYSTEM_PROMPT

    system_message += "\n\nUSER MEMORY:\n"
    system_message += memory_context

    if rag_context:

        system_message += (
            "\n\nUPLOADED DOCUMENT CONTEXT:\n"
            + rag_context
        )

    messages = [
        {
            "role": "system",
            "content": system_message,
        }
    ]

    # Limit conversation history to keep the app fast.
    messages.extend(
        st.session_state.messages[-12:]
    )

    messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    try:

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.4,
            max_tokens=700,
        )

        assistant_message = response.choices[0].message

        # ----------------------------------------------------
        # TOOL CALLING
        # ----------------------------------------------------

        if assistant_message.tool_calls:

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tool.id,
                            "type": "function",
                            "function": {
                                "name": tool.function.name,
                                "arguments": tool.function.arguments,
                            },
                        }
                        for tool in assistant_message.tool_calls
                    ],
                }
            )

            for tool_call in assistant_message.tool_calls:

                name = tool_call.function.name

                arguments = json.loads(
                    tool_call.function.arguments
                )

                result = execute_tool(
                    name,
                    arguments
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

            final_response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.4,
                max_tokens=700,
            )

            return (
                final_response.choices[0]
                .message
                .content
                .strip()
            )

        return (
            assistant_message.content or
            "I couldn't generate a response."
        ).strip()

    except Exception as e:

        error_text = str(e)

        if "401" in error_text or "authentication" in error_text.lower():
            return (
                "Your Groq API key appears to be invalid. "
                "Please check the key in the sidebar."
            )

        if "rate" in error_text.lower():
            return (
                "The Groq API rate limit was reached. "
                "Please wait a moment and try again."
            )

        return f"I encountered an AI error: {error_text}"


# ============================================================
# BROWSER TEXT TO SPEECH
# ============================================================

def speak_text(text):

    if not text:
        return

    # Prevent excessively large browser speech payloads.
    text = text[:2500]

    escaped = json.dumps(text)

    html = f"""
    <script>
        const text = {escaped};

        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();

            const utterance =
                new SpeechSynthesisUtterance(text);

            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;

            window.speechSynthesis.speak(utterance);
        }}
    </script>
    """

    st.components.v1.html(
        html,
        height=0,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Agent Settings")

    api_key_from_ui = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help=(
            "Your key is used only for this Streamlit session "
            "and is never hard-coded."
        ),
    )

    api_key = (
        api_key_from_ui.strip()
        if api_key_from_ui.strip()
        else get_api_key()
    )

    if api_key:
        st.success("Groq API key detected.")
    else:
        st.warning(
            "Enter a Groq API key or configure "
            "GROQ_API_KEY in Streamlit Secrets."
        )

    st.divider()

    st.markdown("### 🧠 Memory")

    if st.session_state.memory:

        for item in st.session_state.memory:

            st.markdown(
                f"""
                <div class="memory-card">
                    {item}
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        st.caption(
            "No long-term information remembered yet."
        )

    if st.button(
        "🗑️ Clear Memory",
        use_container_width=True
    ):

        st.session_state.memory = []

        st.rerun()

    st.divider()

    st.markdown("### 📚 Document RAG")

    if not RAG_AVAILABLE:

        st.warning(
            "RAG dependencies are not available."
        )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs and ask Nova questions about them.",
    )

    if st.button(
        "🔎 Index Documents",
        use_container_width=True,
        disabled=not uploaded_files,
    ):

        with st.spinner("Reading and indexing documents..."):

            process_pdfs(uploaded_files)

        if st.session_state.rag_index is not None:

            st.success(
                f"Indexed {len(st.session_state.rag_chunks)} chunks."
            )

    if st.session_state.rag_files:

        st.caption("Indexed documents:")

        for filename in st.session_state.rag_files:

            st.write(f"📄 {filename}")

    st.divider()

    st.markdown("### 🤖 Capabilities")

    st.markdown(
        """
        - 🎙️ Voice input
        - 🧠 Conversation memory
        - 💾 Long-term memory
        - 📚 PDF RAG
        - 🔎 FAISS retrieval
        - 🛠️ Tool calling
        - 🧮 Calculator
        - 🕐 Date/time tool
        - 🔊 Browser voice output
        """
    )

    st.divider()

    if st.button(
        "🔄 New Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎙️ Nova Voice AI Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Real-time voice AI with memory, RAG and tools"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# STATUS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        """
        <div class="status-card">
        🎙️ <b>Voice</b><br>
        <small>Whisper STT</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:

    st.markdown(
        """
        <div class="status-card">
        🧠 <b>Memory</b><br>
        <small>Session + user facts</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:

    st.markdown(
        """
        <div class="status-card">
        📚 <b>RAG</b><br>
        <small>FAISS PDF search</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:

    st.markdown(
        """
        <div class="status-card">
        🛠️ <b>Tools</b><br>
        <small>Agent tool calling</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# API CLIENT
# ============================================================

client = None

if api_key:

    client = create_client(api_key)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] not in ["user", "assistant"]:
        continue

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# VOICE INPUT
# ============================================================

st.markdown("### 🎤 Talk to Nova")

audio_value = st.audio_input(
    "Record your message",
    sample_rate=16000,
    key="voice_input",
)


# ============================================================
# PROCESS VOICE INPUT
# ============================================================

if audio_value is not None:

    if not api_key:

        st.error(
            "Please enter your Groq API key first."
        )

    elif client is None:

        st.error(
            "Groq client could not be initialized."
        )

    else:

        audio_id = hash(
            audio_value.getvalue()
        )

        if (
            audio_id !=
            st.session_state.last_audio_id
        ):

            st.session_state.last_audio_id = audio_id

            with st.spinner(
                "🎧 Listening and transcribing..."
            ):

                transcript = transcribe_audio(
                    client,
                    audio_value,
                )

            if transcript:

                st.info(
                    f"**You said:** {transcript}"
                )

                detect_memory(transcript)

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": transcript,
                    }
                )

                with st.spinner(
                    "🤖 Nova is thinking..."
                ):

                    answer = generate_response(
                        client,
                        transcript,
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                st.rerun()


# ============================================================
# TEXT INPUT
# ============================================================

text_prompt = st.chat_input(
    "Or type a message..."
)

if text_prompt:

    if not api_key:

        st.error(
            "Please enter your Groq API key first."
        )

    elif client is None:

        st.error(
            "Groq client could not be initialized."
        )

    else:

        detect_memory(text_prompt)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": text_prompt,
            }
        )

        with st.spinner(
            "🤖 Nova is thinking..."
        ):

            answer = generate_response(
                client,
                text_prompt,
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        # Browser TTS
        speak_text(answer)

        st.rerun()


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:60px 20px;
            color:#94a3b8;
        ">

        <div style="font-size:70px;">🎙️</div>

        <h2 style="color:#f8fafc;">
            Meet Nova
        </h2>

        <p>
            Record your voice or type a message to start.
        </p>

        <p>
            Upload PDFs to give Nova a knowledge base.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )
```
