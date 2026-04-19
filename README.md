# ⚖️ Legal RAG Assistant

A Retrieval-Augmented Generation (RAG) system for answering legal questions using **Supreme Court of India judgments** with grounded reasoning and citation support.

---

## 🚀 Overview

This project builds a **legal question-answering system** that retrieves relevant case law and generates answers grounded in actual judicial decisions.

Unlike generic LLMs, this system:
- Uses **real court judgments as context**
- Provides **case-based reasoning** with citations
- Minimizes hallucinations through **retrieval + reranking + evaluation**

---

## 🔐 How to Run

Create a `.env` file:

Or set environment variable:

```bash
export OLLAMA_API_KEY=your_api_key_here
```
you can get your api_key after logging in from [here](https://ollama.com/settings/keys)

Run from terminal
```
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Key Features

- 🔍 **Semantic Retrieval (FAISS)**  
  Efficient similarity search over judgment chunks using dense embeddings  

- 🧩 **Chunked Legal Documents**  
  Judgments split into meaningful segments for better context retrieval  

- ⚖️ **Reranking (Cross-Encoder)**  
  Improves relevance of retrieved chunks using a cross-encoder model  

- 🤖 **LLM-based Answer Generation**  
  Answers generated strictly from retrieved context  

- 📚 **Grounded Citations**  
  Inline citations with full case names and dates  

- 🧪 **Evaluation Framework**
  - LLM-based evaluation (grounding, completeness, hallucination)
  - Citation verification (primary / secondary / hallucinated)

- 🚫 **Out-of-domain Detection**
  Rejects non-legal queries to prevent hallucinated responses  

---

## 📂 Dataset

- **Source:** Indian Kanoon (public domain, [Kaggle](https://www.kaggle.com/datasets/adarshsingh0903/legal-dataset-sc-judgments-india-19502024))
- **Content:** Supreme Court Judgments (1950–2024)
- **Size:** ~400 processed documents (extendable)
- **Format:** PDF → cleaned text → chunked

---

## 🏗️ System Architecture
---

## ⚙️ Tech Stack

- **Python**
- **FAISS** (vector search)
- **Sentence Transformers** (embeddings)
- **Cross-Encoder (MS MARCO)** (reranking)
- **Ollama** (LLM inference)
- **Streamlit** (UI)

---

## 🖥️ Demo UI

Features:
- Query input
- Generated legal answer
- Evaluation metrics
- Source case chunks

---

## 📊 Evaluation Metrics

Each response is evaluated using:

- **Grounding**: Is the answer supported by context?
- **Completeness**: Does it fully answer the query?
- **Hallucination**: Any unsupported claims?
- **Citation Metrics**:
  - Primary (direct case match)
  - Secondary (context match)
  - Hallucinated

---

## 🧪 Example Queries

### ✅ In-domain
- What is the effect of delay in FIR?
- When can investigation be transferred to CBI?
- Can a writ petition be dismissed due to lack of legal right?

### ⚠️ Partial / Out-of-context
- What is anticipatory bail?
- What is Section 482 CrPC?

### ❌ Rejected
- What is reinforcement learning?
- Explain neural networks