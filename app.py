import streamlit as st
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

if query:

    with st.spinner("Understanding query..."):
        # # ---- Step 1: Rewrite ----
        rewritten_query = rewrite_query(query)
        answer = ""

        if rewritten_query.strip() == "Not a legal query":
            st.error("❌ Not a legal query")

            answer = "REJECTED"
            result = {}

    if answer != "REJECTED":
        with st.spinner("Processing..."):        

            # ---- Step 2: Retrieval ----
            chunks = retrieve_with_rerank(rewritten_query)
            chunks = merge_same_case(chunks)

            if len(chunks) == 0:
                st.warning("No relevant legal context found.")

            # ---- Step 3: Generate Answer ----
            answer = generate_answer(rewritten_query, chunks)

            # ---- Step 4: Evaluate ----
            result = evaluate_single(query, {}, answer, chunks)

        # ==============================
        # OUTPUT
        # ==============================

        # ---- Answer ----
        st.subheader("📜 Answer")
        st.write(answer)

        # ---- Evaluation ----
        st.subheader("🧠 Evaluation")

        col1, col2, col3, col4 = st.columns(4)

        llm_eval = result.get("llm_eval", {})

        col1.metric("Grounding", llm_eval.get("grounding", "N/A"))
        col2.metric("Completeness", llm_eval.get("completeness", "N/A"))
        col3.metric("Hallucination", llm_eval.get("hallucination", "N/A"))
        col4.metric("Score", llm_eval.get("score (out of 5)", "N/A"))

        st.write("### Citation Metrics")
        col1, col2, col3 = st.columns(3, gap='xxsmall')

        col1.write(f"**Primary:** {result.get('primary_citation_score', 0.0):.2f}")
        col2.write(f"**Secondary:** {result.get('secondary_citation_score', 0.0):.2f}")
        col3.write(f"**Hallucination:** {result.get('hallucination_rate', 0.0):.2f}")

        # ---- Sources ----
        st.subheader("📚 Sources")

        for i, chunk in enumerate(chunks):
            with st.expander(f"{chunk['case_name']}"):
                st.write(chunk["text"])
            
st.divider()

st.caption("Built using FAISS + Sentence Transformers + Cross-Encoder Reranking + LLM Evaluation  " \
"Dataset: Supreme Court of India Judgments (1950-2024)")