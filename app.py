import streamlit as st
import time
from helper import rewrite_query, retrieve_with_rerank, merge_same_case, generate_answer, evaluate_single, get_resources


@st.cache_resource
def load_resources():
    return get_resources()
resources = load_resources()

st.set_page_config(page_title="Legal RAG Assistant", layout="wide")

st.title("⚖️ Legal RAG Assistant")

st.markdown("""
Uses **Retrieval-Augmented Generation (RAG)** on a curated dataset of  **Supreme Court of India Judgments** to provide legally grounded answers with proper citations.  
⚠️ *This is a research/demo system and not a substitute for professional legal advice.*
""")

st.divider()

# ---- Input ----
query = st.text_input("Ask a legal question:")
enable_answer_rewrite = st.checkbox("Enable Query Rewriting (for better retrieval but slow)")
enable_llm_eval = st.checkbox("Enable LLM Evaluation (slower)")

if query:

    timings = {}

    # ---- Step 1: Rewrite ----
    with st.spinner("Understanding query..."):
        start = time.time()
        if enable_answer_rewrite:
            rewritten_query = rewrite_query(query)
        else:
            rewritten_query = query
        timings["rewrite"] = time.time() - start

        answer = ""

        if rewritten_query.strip() == "Not a legal query":
            st.error("❌ Not a legal query")
            answer = "REJECTED"
            result = {}

    if answer != "REJECTED":

        total_start = time.time()

        with st.spinner("Processing..."):

            # ---- Step 2: Retrieval ----
            start = time.time()
            chunks = retrieve_with_rerank(rewritten_query)
            timings["retrieval"] = time.time() - start

            # ---- Step 3: Merge ----
            start = time.time()
            chunks = merge_same_case(chunks)
            timings["merge"] = time.time() - start

            if len(chunks) == 0:
                st.warning("No relevant legal context found.")

            # ---- Step 4: Generation ----
            start = time.time()
            answer = generate_answer(rewritten_query, chunks)
            timings["generation"] = time.time() - start

            # ---- Step 5: Evaluation ----
            start = time.time()
            result = evaluate_single(query, answer, chunks, enable_llm_eval)
            timings["evaluation"] = time.time() - start

        timings["total"] = time.time() - total_start

        # ==============================
        # OUTPUT
        # ==============================

        # ---- Answer ----
        st.subheader("📜 Answer")
        st.write(answer)

        # ---- Evaluation ----
        if enable_llm_eval:
            st.subheader("🧠 Evaluation")

            col1, col2, col3, col4 = st.columns(4)

            llm_eval = result.get("llm_eval", {})

            col1.metric("Grounding", llm_eval.get("grounding", "N/A"))
            col2.metric("Completeness", llm_eval.get("completeness", "N/A"))
            col3.metric("Hallucination", llm_eval.get("hallucination", "N/A"))
            col4.metric("Score (out of 5)", llm_eval.get("score", "N/A"))

        st.write("### Citation Metrics")
        col1, col2, col3 = st.columns(3, gap='xxsmall')

        col1.write(f"**Primary:** {result.get('primary_citation_score', 0.0):.2f}")
        col2.write(f"**Secondary:** {result.get('secondary_citation_score', 0.0):.2f}")
        col3.write(f"**Hallucination:** {result.get('hallucination_rate', 0.0):.2f}")

        st.subheader("⏱️ Performance Metrics")

        col1, col2, col3 = st.columns(3)

        col1.write(f"Rewrite: {timings.get('rewrite', 0):.2f}s")
        col1.write(f"Retrieval: {timings.get('retrieval', 0):.2f}s")

        col2.write(f"Merge: {timings.get('merge', 0):.2f}s")
        col2.write(f"Generation: {timings.get('generation', 0):.2f}s")

        col3.write(f"Evaluation: {timings.get('evaluation', 0):.2f}s")
        col3.write(f"Total: {timings.get('total', 0):.2f}s")

        # ---- Sources ----
        st.subheader("📚 Sources")

        for i, chunk in enumerate(chunks):
            with st.expander(f"{chunk['case_name']}"):
                st.write(chunk["text"])
            
st.divider()

st.caption("Built using FAISS + Sentence Transformers + Cross-Encoder Reranking + LLM Evaluation  " \
"Dataset: Supreme Court of India Judgments (1950-2024)")