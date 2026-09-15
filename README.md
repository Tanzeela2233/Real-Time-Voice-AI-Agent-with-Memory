````markdown
# 🎙️ Nova — Real-Time Voice AI Agent

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/Groq-AI-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/GenAI-Voice%20Agent-purple?style=for-the-badge">

</p>

<p align="center">
  <b>A real-time AI voice assistant with memory, PDF RAG, tool calling and intelligent conversations.</b>
</p>

<p align="center">

  🚀 <a href="https://real-time-voice-ai-agent123.streamlit.app/">
    <b>Try Nova Live</b>
  </a>

</p>

---

## ✨ Overview

**Nova** is a real-time AI Voice Agent built with **Python and Streamlit**.

Unlike a basic chatbot, Nova combines:

🎙️ Speech-to-Text  
🤖 LLM-powered conversations  
🔊 Text-to-Speech  
🧠 Conversation Memory  
💾 User Information Memory  
📚 PDF RAG  
🔎 FAISS Vector Search  
🛠️ AI Tool Calling  
🧮 Calculator Tool  
🕐 Date & Time Tool  

The project is designed to demonstrate how modern **Generative AI, RAG, voice interaction and agentic capabilities** can be combined into a single lightweight application.

---

## 🚀 Live Demo

### 🎙️ [Launch Nova →](https://real-time-voice-ai-agent123.streamlit.app/)

> Enter your Groq API key through the application or configure it through Streamlit Secrets.

---

## 🧠 Key Features

| Feature | Description |
|---|---|
| 🎙️ Voice Input | Record your voice directly from the browser |
| 📝 Speech-to-Text | Converts speech using Groq Whisper |
| 🤖 AI Brain | Powered by a Groq-supported LLM |
| 🔊 Voice Output | Browser-based SpeechSynthesis |
| 🧠 Conversation Memory | Remembers the current conversation |
| 💾 Long-Term Memory | Stores useful user information during the session |
| 📚 PDF RAG | Upload PDFs and ask questions |
| 🔎 FAISS | Fast semantic document retrieval |
| 🛠️ Tool Calling | AI can decide when to use tools |
| 🧮 Calculator | Performs mathematical calculations |
| 🕐 Date & Time | Provides current date/time through a tool |
| 🔐 Secure API Key | No API key is hard-coded |
| ☁️ Cloud Ready | Deployed with Streamlit Community Cloud |

---

## 🏗️ Architecture

```text
             🎙️ User Voice
                  │
                  ▼
        ┌───────────────────┐
        │ Streamlit Interface│
        └─────────┬─────────┘
                  │
                  ▼
          🎧 Groq Whisper
                  │
                  ▼
             📝 Text
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
     🧠 Memory   📚 RAG   🛠️ Tools
        │         │         │
        └─────────┼─────────┘
                  ▼
             🤖 Groq LLM
                  │
                  ▼
            💬 AI Response
                  │
                  ▼
          🔊 Browser TTS
                  │
                  ▼
             👤 User
````

---

## 🛠️ Tech Stack

### Frontend

* **Streamlit**

### AI / LLM

* **Groq API**
* **OpenAI GPT-OSS model**
* **Groq Whisper**

### RAG

* **FAISS**
* **Sentence Transformers**
* **PyPDF**

### Voice

* **Speech-to-Text:** Groq Whisper
* **Text-to-Speech:** Browser SpeechSynthesis API

### Language

* **Python**

### Deployment

* **GitHub**
* **Streamlit Community Cloud**

---

## 📚 RAG Pipeline

Nova can work with uploaded PDF documents.

```text
📄 PDF
  ↓
Text Extraction
  ↓
Chunking
  ↓
Embeddings
  ↓
FAISS Vector Index
  ↓
Similarity Search
  ↓
Relevant Context
  ↓
🤖 Groq LLM
  ↓
Answer
```

Upload research papers, notes, documentation or other PDFs and ask Nova questions about their contents.

---

## 🛠️ Agent Tools

Nova includes simple tools that can be automatically selected by the AI.

### 🧮 Calculator

```text
"What is 456 × 32?"
```

Nova can call the calculator tool and use its result.

### 🕐 Date & Time

```text
"What is the current date and time?"
```

Nova can use the date/time tool before generating the response.

---

## 🧠 Memory

Nova has two types of memory:

### Conversation Memory

Keeps the current conversation context so follow-up questions feel natural.

### User Memory

Can remember useful information. 

The information can be used in later messages during the active session.

---

## 📁 Project Structure

```text
Real-Time-Voice-AI-Agent/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

The project intentionally keeps the architecture small and easy to understand.

---

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud** directly from GitHub.

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
🚀 Live AI Voice Agent
```

Streamlit Community Cloud automatically deploys changes pushed to the connected repository.

---

## 🎯 Why This Project?

This project demonstrates practical Generative AI concepts in one application:

* Large Language Models
* Speech AI
* Retrieval-Augmented Generation
* Vector Databases
* Embeddings
* Agent Tool Calling
* Memory
* Prompt Engineering
* Streamlit Application Development
* Cloud Deployment

It is designed as a **portfolio-ready GenAI project** while keeping the architecture simple enough for a solo developer.

---

## 🔮 Future Improvements

* 🌐 Web Search Agent
* 🗃️ Persistent Database Memory
* 👤 User Authentication
* ⚡ Streaming AI Responses
* 🎤 Advanced Voice Activity Detection
* 🌍 Urdu / Multilingual Voice Support
* 📅 Calendar Integration
* 📧 Email Agent
* 🔗 More External Tools

---
