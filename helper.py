import json
import re
import faiss
import json
import config
import unicodedata
import numpy as np
from tqdm import tqdm
from ollama import Client
from sentence_transformers import SentenceTransformer, CrossEncoder

_resources = None

def load_chunks_json(file_path):
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def get_resources():
    global _resources

    if _resources is None:
        print("Loading models...")

        all_chunks = load_chunks_json(config.chunk_json_path)
        reranker = CrossEncoder(config.crossencoder_model_name)
        model = SentenceTransformer(config.sentence_transformer_model_name)

        embeddings = np.load(config.embeddings_path)
        dim = embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        client = Client(
            host='https://ollama.com',
            headers={'Authorization': 'Bearer ' + config.ollama_api_key}
        )

        _resources = {
            "all_chunks": all_chunks,
            "reranker": reranker,
            "model": model,
            "index": index,
            "client": client
        }

    return _resources

def retrieve(query, k=5):
    res = get_resources()
    model = res["model"]
    all_chunks = res["all_chunks"]
    index = res["index"]

    q_emb = model.encode([query]).astype("float32")
    
    distances, indices = index.search(q_emb, k)
    
    results = [all_chunks[i] for i in indices[0]]
    return results

def rerank(query, retrieved_chunks, top_k=5):
    pairs = [(query, chunk["text"]) for chunk in retrieved_chunks]

    reranker = get_resources()['reranker']    
    scores = reranker.predict(pairs)
    
    ranked = sorted(zip(scores, retrieved_chunks), key=lambda x: x[0], reverse=True)
    
    return [chunk for _, chunk in ranked[:top_k]]

def retrieve_with_rerank(query):
    initial = retrieve(query, k=20)
    final = rerank(query, initial, top_k=5)
    return final

def build_context(chunks):
    context = ""
    for i, c in enumerate(chunks):
        context += f"[Source {i+1}]\n{c['text']}\n\n"
    return context

def build_prompt(query, context):
    return f"""
You are a legal assistant.

Answer the question using ONLY the context provided below.

Context:
{context}

Question:
{query}

Instructions:
- Be concise and precise
- Use only the provided context
- Always cite sources inline strictly in this format: [Amit Kumar vs Union of India, 24 Mar 2025]. 
- Use nuanced legal language (avoid absolute terms)
- Do NOT use "supra", "ibid", or shortened references.
- If context is partially sufficient, answer what is supported and briefly note missing aspects
- Do NOT introduce new case names or legal provisions not present in the context.
- Use multiple cases to support reasoning where possible, not just one dominant case.

Structure:
1. Legal Principle (3-5 lines)
2. Key Reasoning from Cases (3-5 lines, with citations)
3. Conclusion (short)

At the end, provide:
Confidence: High / Medium / Low based on completeness of context

Answer:
"""

def call_ollama_cloud(prompt, model="gpt-oss:120b"):
    client = get_resources()['client']
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response['message']['content']

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    
    text = text.replace("\u200b", "")   # zero-width space
    text = text.replace("\u202f", " ")  # narrow no-break space
    
    return text

def generate_answer(query, chunks, model="gpt-oss:120b"):
    
    if not chunks:
        return "No relevant legal context found to answer the query."

    context = ""
    for c in chunks:
        context += (
            f"{c['text']}\n\n"
        )

    prompt = build_prompt(query, context)
    answer = call_ollama_cloud(prompt, model)
    answer_clean = normalize_text(answer)

    return answer_clean

def rewrite_query(query):
    prompt = f"""
You are a classifier and rewriter.

Task:
1. Determine if the query is related to law, legal systems, courts, crime, rights, or legal procedures.
2. If YES:
   - Rewrite it into a clear and concise legal query (Indian legal context if applicable).
   - Keep it short (1 line).
3. If NO:
   - Return exactly: Not a legal query

Rules:
- Be permissive: if the query could reasonably relate to law or legal concepts, treat it as legal.
- Queries about crime, police, FIR, courts, evidence, rights, or procedures are ALL legal.
- Do NOT overthink or reject borderline queries.

Query:
{query}

Output:
"""
    return call_ollama_cloud(prompt)

def diversify_chunks(chunks):
    seen_cases = dict()
    unique = []

    for c in chunks:
        if c["case_name"] not in seen_cases:
            unique.append(c)
            seen_cases[c["case_name"]] = 1
        else:
            seen_cases[c["case_name"]] += 1
        if seen_cases[c["case_name"]] <= 3:
            unique.append(c)

    return unique

def merge_same_case(chunks):
    merged = {}
    
    for c in chunks:
        key = c["case_name"]
        if key not in merged:
            merged[key] = c
        else:
            merged[key]["text"] += " " + c["text"]
    
    return list(merged.values())

def is_citation_in_context(citation, context):
    words = citation.lower().split()
    
    match_count = sum(1 for w in words if w in context)
    
    return match_count >= max(2, len(words) // 2)

def classify_citations(answer, chunks):
    context = " ".join([c["text"] for c in chunks]).lower()
    case_names = [c["case_name"].lower() for c in chunks]

    cited = re.findall(r'\[(.*?)\]', answer)
    primary, secondary, hallucinated = 0, 0, 0

    for c in cited:
        c_low = c.lower()

        if any(name in c_low for name in case_names):
            primary += 1
        elif c_low in context:
            secondary += 1
        else:
            hallucinated += 1

    return primary, secondary, hallucinated

def evaluate_single(query, item, answer, chunks):
    # ---- Build context ----
    context = " ".join([c["text"] for c in chunks]).lower()
    case_names = [c["case_name"].lower() for c in chunks]

    # ---- Extract citations ----
    cited = set(re.findall(r'\[\s*(.*?)\s*\]', answer))

    primary, secondary, hallucinated = 0, 0, 0

    for c in cited:
        c_low = c.lower()

        if any(name in c_low for name in case_names):
            primary += 1
        elif is_citation_in_context(c_low, context):
            secondary += 1
        else:
            hallucinated += 1

    total = len(cited)

    if total > 0:
        primary_score = primary / total
        secondary_score = secondary / total
        hallucination_rate = hallucinated / total
    else:
        primary_score = 0
        secondary_score = 0
        hallucination_rate = 0

    # ---- LLM evaluation ----
    try:
        llm_eval_raw = llm_evaluate(query, context, answer)
        llm_eval = json.loads(llm_eval_raw)
    except:
        llm_eval = {"error": "invalid_json"}

    return {
        "query": query,

        # ---- LLM judgment ----
        "llm_eval": llm_eval,

        # ---- Citation metrics ----
        "num_citations": total,
        "primary_citation_score": primary_score,
        "secondary_citation_score": secondary_score,
        "hallucination_rate": hallucination_rate,

        # ---- Debugging ----
        "citations_extracted": list(cited),
        "answer_preview": answer[:200]
    }

def llm_evaluate(query, context, answer):

    if not context:
        return json.dumps({})

    prompt = f"""
    You are an expert legal evaluator.

    Evaluate strictly using ONLY the allowed values based on the provided context and the answer.

    Grounding:
    - Allowed values: Yes / Partial / No

    Completeness:
    - Allowed values: Complete / Partial / Incomplete

    Hallucination:
    - Allowed values: None / Minor / Major

    Return EXACTLY this format:
    {{
    "grounding": "Yes/Partial/No",
    "completeness": "Complete/Partial/Incomplete",
    "hallucination": "None/Minor/Major",
    "score": 1-5
    }}

    Question:
    {query}

    Context:
    {context}

    Answer:
    {answer}
    """

    response = call_ollama_cloud(prompt)

    return response

def run_full_evaluation(evaluation_set):

    results = []

    for item in tqdm(evaluation_set):
        query = item["query"]

        # ---- Step 1: Rewrite / classify ----
        rewritten_query = rewrite_query(query)

        # ---- Step 2: Handle rejection ----
        if rewritten_query.strip() == "Not a legal query":
            answer = "REJECTED"
            chunks = []

        else:
            # ---- Step 3: Retrieval ----
            chunks = retrieve_with_rerank(rewritten_query)

            # ---- Step 4: Post-process ----
            chunks = merge_same_case(chunks)

            # ---- Step 5: Generation ----
            answer = generate_answer(rewritten_query, chunks)

        # ---- Step 6: Evaluation ----
        result = evaluate_single(query, item, answer, chunks)
        results.append(result)

    return results


