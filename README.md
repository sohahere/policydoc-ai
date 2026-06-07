
# PolicyDoc AI

<p align="center">
  <img src="assets/policydoc_ai_project_overview_banner.png" alt="PolicyDoc AI Project Banner" width="100%">
</p>

## Evaluated RAG Assistant for Company Policy & Compliance Documents with LLMOps-Lite Observability

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue" />
  <img src="https://img.shields.io/badge/Streamlit-UI-red" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20DB-green" />
  <img src="https://img.shields.io/badge/Sentence%20Transformers-Embeddings-purple" />
  <img src="https://img.shields.io/badge/Hugging%20Face-LLM-yellow" />
  <img src="https://img.shields.io/badge/Docker-Supported-blue" />
  <img src="https://img.shields.io/badge/Hugging%20Face%20Spaces-Deployed-orange" />
  <img src="https://img.shields.io/badge/RAG-Evaluated-success" />
  <img src="https://img.shields.io/badge/LLMOps--Lite-Observability-orange" />
</p>

PolicyDoc AI is an end-to-end Retrieval-Augmented Generation system built for company policy and compliance documents. It allows users to upload TXT or PDF policy documents, extract text and tables, convert tables into markdown, store document chunks in ChromaDB, retrieve relevant evidence, and generate grounded answers with source citations.

Unlike a basic “chat with PDF” project, PolicyDoc AI includes table-aware ingestion, evaluated retrieval, hallucination control, Streamlit UI, Docker support, advanced retrieval experiments, prompt versioning, structured logging, custom exceptions, and lightweight LLMOps-style run tracing.

---

## Live Demo & Deployment

- **Live Hugging Face Spaces App:** https://huggingface.co/spaces/sohahere/policydoc-ai
- **Direct App URL:** https://sohahere-policydoc-ai.hf.space
- **GitHub Repository:** https://github.com/sohahere/policydoc-ai
- **Deployment Type:** Docker-based Hugging Face Space running a Streamlit RAG application on port `7860`

> Note: The deployed app requires `HF_TOKEN` to be configured as a Hugging Face Space secret. Locally, the same token should be stored in a `.env` file.

---

## Problem Statement

Companies store important operational knowledge inside long policy and compliance documents such as employee handbooks, leave policies, expense reimbursement policies, access-control policies, security incident reporting documents, and retention manuals.

These documents often contain dense paragraphs, tables, exceptions, approval rules, and page-specific clauses. Searching manually through them is slow and error-prone.

Employees may ask questions like:

```txt
What is the local meal reimbursement limit?
How many remote work days are allowed per week?
Who can access restricted documents?
How quickly should a lost laptop be reported?
What is the customer refund policy?
```

A normal LLM may answer from general knowledge and hallucinate. In policy and compliance use cases, this is risky because answers must be grounded in the actual company document.

PolicyDoc AI solves this using Retrieval-Augmented Generation:

```txt
User question
   ↓
Retrieve relevant policy evidence from uploaded documents
   ↓
Generate an answer only from retrieved context
   ↓
Show source citations
```

If the uploaded document does not contain the answer, the system returns an insufficient-context response instead of guessing.

---

## Demo Screenshots

### 1. Upload and Index Policy Document

The user can upload a company policy PDF or TXT file and index it into ChromaDB.

<p align="center">
  <img src="assets/streamlit_upload_index.png" alt="Streamlit Upload and Index Screenshot" width="90%">
</p>

### 2. Policy Question Answering with Citations

Example question:

```txt
What is the local meal reimbursement limit?
```

The system retrieves relevant policy evidence and generates an answer with citations.

<p align="center">
  <img src="assets/streamlit_policy_answer.png" alt="Policy Answer with Citations Screenshot" width="90%">
</p>

### 3. Negative Question / Missing Policy Handling

Example question:

```txt
What is the customer refund policy?
```

Since the uploaded document does not contain a customer refund policy, the system refuses to guess.

<p align="center">
  <img src="assets/streamlit_negative_answer.png" alt="Negative Query Handling Screenshot" width="90%">
</p>

---

## Key Features

| Feature | Description |
|---|---|
| **TXT/PDF ingestion** | Supports text files and text-based PDF policy documents. |
| **PDF text extraction** | Extracts page-wise text from PDF documents. |
| **PDF table extraction** | Extracts tables separately instead of treating them as noisy text. |
| **Table-to-markdown conversion** | Converts extracted tables into markdown format to preserve header-value relationships. |
| **Metadata-based citations** | Stores file name, page number, content type, table index, chunk ID, and chunk length. |
| **Evaluated chunking strategy** | Uses tested chunk size and overlap instead of random values. |
| **Semantic retrieval** | Uses sentence-transformer embeddings and ChromaDB for vector search. |
| **LLM answer generation** | Uses a Hugging Face LLM to generate grounded answers from retrieved context. |
| **Source citations** | Shows which file, page, chunk, and content type were used for the answer. |
| **Negative-question handling** | Returns an insufficient-context response when the answer is not present. |
| **Streamlit UI** | Provides an interactive interface for uploading documents and asking questions. |
| **Docker support** | Includes Docker setup for containerized local deployment. |
| **Hugging Face Spaces deployment** | Publicly deployed as a Docker-based Streamlit app with secure secret-based `HF_TOKEN` handling. |
| **Advanced retrieval experiments** | Compares dense retrieval, reranking, and hybrid retrieval on a harder benchmark. |
| **Prompt versioning** | Stores the RAG prompt separately in `prompts/policy_rag_prompt_v1.txt`. |
| **Structured logging** | Uses a centralized logger for clean timestamped pipeline logs. |
| **Custom exceptions** | Adds project-specific errors for retrieval, vector store, prompt, and LLM failures. |
| **LLMOps-lite tracing** | Logs RAG runs with app version, prompt version, latency, estimated tokens, model settings, and retrieved source metadata. |

---

## Why This Is More Than a Basic PDF Chatbot

| Basic PDF Chatbot | PolicyDoc AI |
|---|---|
| Usually supports only simple PDF text | Supports TXT, PDF text, and PDF tables |
| Often ignores tables | Extracts tables and converts them to markdown |
| Uses random chunk size | Uses evaluated chunking strategy |
| Retrieves chunks without evaluation | Evaluates retrieval using Recall@3 and MRR |
| May hallucinate missing answers | Handles missing-policy questions safely |
| Often has no citations | Provides metadata-based source citations |
| Usually has no failure analysis | Includes hard benchmark and retrieval experiments |
| Demo only | Streamlit UI + Docker + public Hugging Face Spaces deployment |

---

## Architecture

PolicyDoc AI follows a complete RAG pipeline. The system converts uploaded policy documents into searchable chunks, stores them in ChromaDB, retrieves relevant evidence, and generates an answer with citations.

<p align="center">
  <img src="assets/policydoc_ai_system_architecture_flowchart.png" alt="PolicyDoc AI System Architecture" width="95%">
</p>

### High-Level Flow

```txt
User uploads policy PDF/TXT
        ↓
Document Loader
        ↓
PDF text extraction + PDF table extraction
        ↓
Table-to-markdown conversion
        ↓
LangChain Document objects with metadata
        ↓
Recursive chunking
        ↓
Sentence-transformer embeddings
        ↓
ChromaDB vector database
        ↓
User asks a question
        ↓
Top-k retrieval
        ↓
Load versioned prompt template
        ↓
Hugging Face LLM answer generation
        ↓
Grounded answer with source citations
        ↓
LLMOps-lite trace logging
```

### RAG Pipeline Flow

<p align="center">
  <img src="assets/policydoc_ai_pipeline_flow_chart.png" alt="PolicyDoc AI RAG Pipeline Flow" width="95%">
</p>

The pipeline has two major stages.

### 1. Indexing Stage

```txt
Document → text/table extraction → chunking → embeddings → ChromaDB
```

Each searchable chunk stores metadata:

```txt
file_name
page_number
content_type
table_index
chunk_id
chunk_length
```

### 2. Query Stage

```txt
Question → query embedding → semantic retrieval → versioned prompt → LLM answer → citations → run trace
```

The final answer is generated only from retrieved context. If the answer is missing from the document, the system returns an insufficient-context response instead of guessing.

### Why an Explicit Modular Pipeline Instead of LCEL Chains?

PolicyDoc AI uses LangChain components such as `Document` objects and `RecursiveCharacterTextSplitter`, but the full RAG flow is implemented as an explicit modular pipeline instead of a compact LCEL chain.

This was intentional.

The project needs visibility into each step:

- document loading
- PDF table extraction
- chunk metadata
- embedding creation
- ChromaDB storage
- retrieved chunk IDs
- distance scores
- citation metadata
- evaluation reports
- prompt versioning
- LLMOps run logs

A formal chain can reduce code length, but it can also hide intermediate outputs. Since this project focuses on evaluation, debugging, and explainability, the pipeline is kept explicit and modular.

Conceptually, the system is still chained:

```txt
loader → chunker → embeddings → vector store → retriever → prompt builder → LLM → logger
```

In a future production version, these same modules could be wrapped into LCEL chains or service classes.

---

## Document Ingestion and Table Handling

PolicyDoc AI supports two document types:

```txt
.txt
.pdf
```

The ingestion layer is implemented in:

```txt
app/loaders.py
```

### TXT Loading

For `.txt` files, the loader reads the full text and converts it into a LangChain `Document` object with metadata.

```python
{
    "file_name": "policy_notes.txt",
    "file_type": "txt",
    "content_type": "text",
    "page_number": None
}
```

### PDF Loading

For `.pdf` files, the loader extracts page-wise text, tables, page number, content type, and table index.

Example text metadata:

```python
{
    "file_name": "sample.pdf",
    "file_type": "pdf",
    "content_type": "text",
    "page_number": 4
}
```

Example table metadata:

```python
{
    "file_name": "sample.pdf",
    "file_type": "pdf",
    "content_type": "table",
    "page_number": 4,
    "table_index": 1
}
```

### Why Table Extraction Matters

Company policy documents often store important values inside tables: leave allowance, reimbursement limits, access permissions, retention periods, and incident reporting deadlines.

Raw table row:

```txt
Local Meal | USD 25 per person | Required for all claims | Alcohol is not reimbursable
```

Markdown table version:

```md
| Expense Category | Limit | Receipt Required | Notes |
|---|---|---|---|
| Local Meal | USD 25 per person | Required for all claims | Alcohol is not reimbursable |
```

This helps the retrieval system and the LLM understand which value belongs to which policy field.

### Metadata-Based Citations

Every chunk keeps metadata such as:

```txt
file_name
page_number
content_type
table_index
chunk_id
chunk_length
```

Example citation:

```txt
Source 1: file=sample.pdf, page=4, chunk_id=11, content_type=text
Source 2: file=sample.pdf, page=4, chunk_id=13, content_type=table
```

---

## Chunking Strategy

PolicyDoc AI splits extracted text and table content into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`.

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

Chunking is important because:

- LLMs have limited context windows.
- Very large chunks may mix unrelated policy topics.
- Very small chunks may lose important context.
- Retrieval quality depends heavily on chunk quality.

### Why RecursiveCharacterTextSplitter?

It tries to split text using natural boundaries first:

```txt
paragraphs → lines → spaces → characters
```

This is better than blindly cutting text every fixed number of characters because it tries to preserve meaning inside each chunk.

### Why Chunk Overlap?

Policy clauses often continue across sentences or paragraphs. If chunks are split with no overlap, important context may be lost at the boundary.

```txt
Chunk 1: Employees must submit reimbursement claims within 30 days...
Chunk 2: Claims submitted after the deadline may be rejected...
```

Overlap keeps a small part of the previous chunk inside the next chunk, preserving continuity.

### Chunking Configurations Tested

| Config | Chunk Size | Overlap | Purpose |
|---|---:|---:|---|
| Small | 500 | 100 | More focused chunks, but more total chunks |
| Balanced | 800 | 150 | Middle-ground configuration |
| Large | 1000 | 200 | Fewer chunks with more context |

The final selected configuration was `chunk_size=1000` and `chunk_overlap=200`.

---

## Embeddings and Vector Database

After chunking, each chunk is converted into a numerical vector called an embedding.

PolicyDoc AI uses:

```txt
sentence-transformers/all-MiniLM-L6-v2
```

Reasons for choosing this model:

- free and local
- fast enough on CPU
- suitable for semantic search
- does not require paid embedding APIs
- produces 384-dimensional embeddings
- easy to use with `sentence-transformers`

### ChromaDB Vector Store

The project uses local ChromaDB as the vector database. For each chunk, ChromaDB stores:

```txt
1. chunk text
2. embedding vector
3. metadata
```

Example stored item:

```python
{
    "id": "sample_chunk_13",
    "document": "| Expense Category | Limit | Receipt Required | Notes | ...",
    "embedding": [0.021, -0.113, 0.452, ...],
    "metadata": {
        "file_name": "sample.pdf",
        "page_number": 4,
        "chunk_id": 13,
        "content_type": "table"
    }
}
```

During retrieval:

```txt
User question
   ↓
Question converted into embedding
   ↓
ChromaDB compares question embedding with stored chunk embeddings
   ↓
Top-k most similar chunks are returned
```

Local ChromaDB runs with:

```python
chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
```

This means no ChromaDB account is required. For production, local ChromaDB can be replaced with a managed vector database.

---

## Answer Generation and Hallucination Control

After retrieval, the top relevant chunks are passed to a Hugging Face LLM for answer generation.

The RAG prompt is versioned separately from the Python code in:

```txt
prompts/policy_rag_prompt_v1.txt
```

This makes prompt changes easier to track and improves reproducibility.

<p align="center">
  <img src="assets/prompt_versioning.png" alt="Prompt Versioning in PolicyDoc AI" width="90%">
</p>

The prompt tells the model to:

- use only the retrieved context
- avoid outside knowledge
- keep the answer concise
- include source citations
- say when the answer is not present in the document
- mention table content when the answer comes from a table

Grounding instruction:

```txt
Use ONLY the provided context to answer the question.
If the answer is not present in the context, say:
"I could not find this information in the provided documents."
```

### Positive Answer Example

Question:

```txt
What is the local meal reimbursement limit?
```

Answer:

```txt
The local meal reimbursement limit is USD 25 per person.
```

Retrieved sources:

```txt
Source 1: file=sample.pdf, page=4, chunk_id=11, content_type=text
Source 2: file=sample.pdf, page=4, chunk_id=13, content_type=table
```

<p align="center">
  <img src="assets/streamlit_policy_answer.png" alt="Policy answer with citation" width="90%">
</p>

### Negative Question Handling

Question:

```txt
What is the customer refund policy?
```

Response:

```txt
I could not find this information in the provided documents.
```

<p align="center">
  <img src="assets/streamlit_negative_answer.png" alt="Negative question handling" width="90%">
</p>

---

## LLMOps-Inspired Observability

PolicyDoc AI includes a lightweight LLMOps-inspired observability layer to make the RAG system more reproducible, traceable, and easier to debug.

This is not a full production LLMOps platform, but it adds practical tracing and monitoring concepts commonly used in real-world LLM applications.

### What Gets Tracked

| Logged Field | Purpose |
|---|---|
| `timestamp_utc` | Tracks when the query was executed |
| `app_version` | Tracks the application version |
| `prompt_version` | Tracks which prompt template was used |
| `question` | Stores the user query for debugging |
| `answer` | Stores the generated response |
| `latency_seconds` | Measures end-to-end RAG response time |
| `estimated_question_tokens` | Approximate question token count |
| `estimated_answer_tokens` | Approximate answer token count |
| `retrieved_source_count` | Number of retrieved chunks |
| `retrieved_sources` | File name, page number, chunk ID, content type, and vector distance |
| `model_name` | LLM used for answer generation |
| `top_k` | Number of chunks retrieved |
| `temperature` | Generation randomness setting |
| `max_tokens` | Output token limit |

### Example RAG Trace

Each RAG run is logged locally as a JSONL record:

```txt
logs/rag_runs.jsonl
```

Example trace:

<p align="center">
  <img src="assets/llmops_trace_log.png" alt="LLMOps Trace Log Example" width="90%">
</p>

The actual runtime log file is ignored by Git because logs may contain user questions, generated answers, or sensitive document content.

```gitignore
logs/*
!logs/.gitkeep
```

> Token counts are lightweight estimates for local observability, not exact provider billing tokens. In production, exact token usage would be tracked using provider metadata or tokenizer-level counting.

### Why This Matters

This tracing layer helps answer questions such as:

- Which prompt version generated this answer?
- Which model and generation settings were used?
- How long did the RAG response take?
- Which chunks were retrieved?
- Did the answer come from text or table content?
- What source pages supported the answer?
- How can we debug a wrong or slow response?

In production, this type of information would usually be sent to observability tools such as LangSmith, Langfuse, Helicone, Arize Phoenix, OpenTelemetry-based tracing, or a logging database.

---

## Evaluation Results

PolicyDoc AI includes retrieval evaluation instead of relying only on manual testing.

The goal is to check:

```txt
Did the retriever bring the correct evidence into the top-k results?
```

### Clean Company Policy Benchmark

The clean benchmark contains questions such as:

```txt
How many paid time off days are allowed per year?
How many remote work days are allowed per week?
What is the local meal reimbursement limit?
Who can access restricted documents?
How quickly should a lost laptop or token be reported?
```

For each question, I defined the expected source page.

```json
{
  "question": "What is the local meal reimbursement limit?",
  "expected_page_numbers": [4]
}
```

### Metrics Used

| Metric | Meaning |
|---|---|
| **Recall@3** | Whether the expected source page appears in the top 3 retrieved chunks |
| **MRR** | How high the first correct retrieved result appears |

### Clean Benchmark Result

| Benchmark | Recall@3 | MRR |
|---|---:|---:|
| Company policy PDF | 1.00 | 1.00 |

The score does **not** mean the system has universal 100% accuracy. It means that on this controlled company-policy benchmark, the selected ingestion, chunking, embedding, and retrieval setup successfully retrieved the expected source page in the top 3 results.

---

## Advanced Retrieval Experiment

A clean benchmark is useful for validating the main pipeline, but real policy documents can be harder.

To test the system more deeply, I created a harder policy benchmark with repeated policy terms, overlapping reimbursement rules, similar access-control clauses, paraphrased questions, table-heavy evidence, and missing-policy traps.

| User Query Wording | Document Wording |
|---|---|
| lunch | local meal |
| client dinner | client meal |
| missing security token | lost laptop or token |
| normal lunch claim | local meal reimbursement |

### Retrieval Methods Compared

| Method | Description |
|---|---|
| **Dense Retrieval** | ChromaDB semantic search using sentence-transformer embeddings |
| **Dense + Reranking** | ChromaDB retrieves candidates, then a cross-encoder reranker reorders them |
| **Dense + BM25 Hybrid** | Combines semantic retrieval with BM25 keyword retrieval |

### Hard Benchmark Results

| Retrieval Method | Recall@3 | MRR |
|---|---:|---:|
| Dense retrieval only | 0.25 | 0.17 |
| Dense + cross-encoder reranking | 0.25 | 0.25 |
| Dense + BM25 hybrid retrieval | 0.25 | 0.17 |
| Candidate Recall@20 | 0.88 | - |

<p align="center">
  <img src="assets/advanced_retrieval_experiment_benchmark_comparison.png" alt="Advanced Retrieval Experiment Benchmark Comparison" width="95%">
</p>

### Interpretation

Candidate Recall@20 was `0.88`, which means the correct evidence was often present somewhere in the broader top-20 candidate pool. However, the correct evidence was not consistently promoted into the final top-3 results.

Reranking improved MRR slightly:

```txt
0.17 → 0.25
```

But Recall@3 did not improve.

BM25 hybrid retrieval also did not improve the hard benchmark because many difficult questions used paraphrased wording. BM25 helps with exact keyword matching, but it does not automatically solve synonym mismatch.

Example:

```txt
User says: lunch
Document says: local meal
```

### Future Improvements from the Hard Benchmark

1. Query rewriting
2. Domain synonym expansion
3. Reciprocal Rank Fusion
4. Row-level table chunking
5. Stronger embedding models such as BGE or E5
6. Better domain-aware reranking

---

## Streamlit UI

The Streamlit UI supports:

- uploading a PDF or TXT policy document
- indexing the uploaded document into ChromaDB
- asking questions from the uploaded document
- generating grounded answers
- viewing retrieved sources
- inspecting page number, chunk ID, content type, and distance score

The same Streamlit interface is deployed publicly on Hugging Face Spaces:

```txt
https://sohahere-policydoc-ai.hf.space
```

<p align="center">
  <img src="assets/streamlit_upload_index.png" alt="Streamlit Upload and Index" width="90%">
</p>

---

## Docker and Hugging Face Spaces Deployment

The project is containerized so that the same RAG application can run locally or on a cloud-hosted demo environment.

### Local Docker Run

Build the Docker image:

```bash
docker build -t policydoc-ai .
```

Run the Docker container locally:

```bash
docker run -p 7860:7860 --env-file .env policydoc-ai
```

Then open:

```txt
http://localhost:7860
```

<p align="center">
  <img src="assets/docker.png" alt="Docker Running PolicyDoc AI" width="90%">
</p>

### Hugging Face Spaces Deployment

The app is deployed publicly as a Docker-based Hugging Face Space:

```txt
https://huggingface.co/spaces/sohahere/policydoc-ai
```

Direct app URL:

```txt
https://sohahere-policydoc-ai.hf.space
```

For Hugging Face Spaces, the Docker container exposes port `7860`, and the app starts with:

```bash
streamlit run frontend/streamlit_app.py --server.address=0.0.0.0 --server.port=7860
```

The `HF_TOKEN` is stored as a Hugging Face Space secret instead of being committed to the repository.

### Storage Note

The current deployment uses local ChromaDB inside the running container. This is suitable for a demo, but for production the vector database should be persisted using a mounted volume or replaced with a managed vector database.

---

## Installation and Usage

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd policydoc-ai
```

### 2. Create a Virtual Environment

```bash
uv venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

Linux / Mac:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Create `.env` File

```env
HF_TOKEN=your_huggingface_token_here
```

Do not push `.env` to GitHub.

### 5. Index the Sample Policy Document

```bash
python app/index_documents.py
```

### 6. Run Retrieval Evaluation

```bash
python app/evaluation.py
```

### 7. Test RAG Answer Generation

```bash
python app/rag_answer.py
```

### 8. Run the Streamlit App

```bash
streamlit run frontend/streamlit_app.py
```

Open:

```txt
http://localhost:8501
```

Or use the deployed Hugging Face Spaces version:

```txt
https://sohahere-policydoc-ai.hf.space
```

---

## Tech Stack

| Layer | Tool / Library | Why It Is Used |
|---|---|---|
| Programming Language | Python | Main development language |
| UI | Streamlit | Interactive app for uploading documents and asking questions |
| PDF Processing | pdfplumber | Extracts text and tables from PDF files |
| Document Format | LangChain `Document` | Stores page content with metadata |
| Chunking | RecursiveCharacterTextSplitter | Splits documents into meaningful chunks |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Converts text chunks into semantic vectors |
| Vector Database | ChromaDB | Stores embeddings, chunks, and metadata |
| LLM | Hugging Face Inference Provider | Generates final grounded answers |
| Environment Variables | python-dotenv | Loads Hugging Face token securely |
| Evaluation | Recall@3 and MRR scripts | Measures retrieval quality |
| Logging | Python `logging` | Produces timestamped process logs |
| Error Handling | Custom exception classes | Makes failures easier to debug and explain |
| LLMOps-lite | JSONL run tracing | Tracks prompt version, latency, estimated tokens, model settings, and retrieved sources |
| Keyword Retrieval Experiment | rank-bm25 | Tests BM25-based hybrid retrieval |
| Reranking Experiment | CrossEncoder | Tests query-chunk reranking |
| Containerization | Docker | Runs the app inside a container |
| Cloud Demo Deployment | Hugging Face Spaces | Hosts the Dockerized Streamlit RAG app publicly with secret-based token handling |

---

## Important Project Files

| File | Purpose |
|---|---|
| `app/config.py` | Stores project paths, model names, chunk settings, retrieval settings, collection names, prompt path, app version, and log paths |
| `app/logger.py` | Provides centralized timestamped logging for pipeline events |
| `app/exceptions.py` | Defines custom project-specific exceptions for cleaner debugging |
| `app/llmops.py` | Loads versioned prompts, builds context, estimates tokens, and logs RAG runs |
| `app/loaders.py` | Loads TXT/PDF files, extracts PDF text and tables, converts tables to markdown |
| `app/index_documents.py` | Loads documents, chunks them, creates embeddings, and stores them in ChromaDB |
| `app/retrieve.py` | Tests retrieval from ChromaDB for a sample query |
| `app/rag_answer.py` | Performs retrieval, builds the RAG prompt, calls the Hugging Face LLM, and returns answers with sources |
| `app/evaluation.py` | Evaluates retrieval on the clean company-policy benchmark |
| `app/reranker.py` | Contains the cross-encoder reranking logic |
| `app/hybrid_retriever.py` | Contains dense + BM25 hybrid retrieval logic |
| `app/evaluate_hard_base.py` | Evaluates dense retrieval on the hard benchmark |
| `app/evaluate_hard_reranking.py` | Evaluates dense retrieval plus reranking on the hard benchmark |
| `app/evaluate_hard_hybrid.py` | Evaluates dense + BM25 hybrid retrieval on the hard benchmark |
| `app/debug_candidate_recall.py` | Checks whether correct evidence appears in the wider top-20 candidate pool |
| `frontend/streamlit_app.py` | Streamlit user interface |
| `prompts/policy_rag_prompt_v1.txt` | Versioned prompt template for grounded RAG answer generation |
| `logs/.gitkeep` | Keeps the logs folder in Git while ignoring runtime logs |
| `evaluation/eval_questions.json` | Clean benchmark questions |
| `evaluation/hard_eval_questions.json` | Hard benchmark questions |
| `data/uploaded_docs/sample.pdf` | Clean company-policy sample PDF |
| `data/uploaded_docs/hard_evaluation.pdf` | Hard retrieval benchmark PDF |
| `Dockerfile` | Docker container definition for local and Hugging Face Spaces deployment |
| `requirements.txt` | Local Python dependencies |
| `requirements-docker.txt` | Docker-specific dependencies |

---

## Configuration

Important project settings are stored in:

```txt
app/config.py
```

Example configuration:

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "policy_documents"
```

LLMOps-lite configuration:

```python
APP_VERSION = "0.1.0"
PROMPT_VERSION = "policy_rag_prompt_v1"
PROMPT_FILE_PATH = PROMPT_DIR / "policy_rag_prompt_v1.txt"
RAG_RUN_LOG_PATH = LOG_DIR / "rag_runs.jsonl"
```

Hard benchmark settings:

```python
CANDIDATE_K = 20
RERANK_TOP_K = 3
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HYBRID_DENSE_K = 20
HYBRID_BM25_K = 20
HYBRID_FINAL_K = 3
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
```

The Hugging Face token is used for LLM answer generation.

Do not push `.env` to GitHub.

For the deployed Hugging Face Space, `HF_TOKEN` is added through **Settings → Variables and secrets → New secret**.

---

## Local ChromaDB Storage

ChromaDB is used locally through:

```python
chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
```

This creates a local vector database folder:

```txt
chroma_db/
```

The folder is ignored in Git because it can be regenerated by running:

```bash
python app/index_documents.py
```

---

## Limitations

| Limitation | Explanation |
|---|---|
| **No OCR support** | Scanned PDFs and image-only documents are not processed in the current version. |
| **No DOCX support** | Current version supports TXT and PDF only. |
| **Single-user demo design** | The Streamlit app is deployed as a public demo, but it is not designed for authenticated multi-user production usage. |
| **Local ChromaDB** | Vector storage is local/ephemeral in the demo container. For production, a managed vector database or mounted Docker volume would be better. |
| **No authentication** | The app does not include login or user-level document isolation. |
| **Basic table chunking** | Tables are converted to markdown, but row-level table retrieval is not implemented yet. |
| **No query rewriting** | User queries are passed directly to retrieval without synonym expansion or rewriting. |
| **No full RAGAS evaluation yet** | Current evaluation uses custom Recall@3 and MRR scripts. |
| **LLMOps-lite only** | Current tracing is local JSONL-based. Full production LLMOps would require external tracing, dashboards, feedback loops, and alerting. |
| **Approximate token tracking** | Token counts are lightweight estimates, not exact provider billing values. |
| **Free LLM dependency** | Hugging Face free-tier inference may have latency or rate-limit issues. |

---

## Future Work

The hard benchmark revealed useful next-step improvements:

1. OCR for scanned PDFs
2. DOCX support
3. Query rewriting
4. Domain synonym expansion
5. Row-level table chunking
6. Reciprocal Rank Fusion
7. Stronger embedding models
8. RAGAS evaluation
9. FastAPI backend
10. Managed vector database
11. Persistent cloud storage for uploaded documents and vector indexes
12. User-specific document collections
13. Production observability dashboard and user feedback loop

---

## Skills Demonstrated

| Area | Skills Demonstrated |
|---|---|
| Document AI | TXT/PDF ingestion, PDF text extraction, PDF table extraction |
| RAG Engineering | chunking, embeddings, vector search, context retrieval, grounded generation |
| Vector Databases | ChromaDB local vector storage, metadata-based retrieval |
| LLM Integration | Hugging Face LLM API, prompt design, context-grounded answer generation |
| Evaluation | Recall@3, MRR, clean benchmark, hard benchmark, candidate recall analysis |
| Retrieval Optimization | dense retrieval, reranking experiment, BM25 hybrid retrieval experiment |
| UI Development | Streamlit app for upload, indexing, Q&A, and source inspection |
| Deployment Basics | Dockerfile, `.dockerignore`, Hugging Face Spaces deployment, environment variable and secret handling |
| LLMOps-lite | prompt versioning, run tracing, latency logging, estimated token tracking, source trace logging |
| Software Engineering | modular code, centralized configuration, custom exceptions, structured logging |
| Production Thinking | hallucination control, limitations analysis, future improvement planning |

---

## Project Status

| Component | Status |
|---|---|
| TXT/PDF ingestion | Complete |
| PDF text extraction | Complete |
| PDF table extraction | Complete |
| Table-to-markdown conversion | Complete |
| Metadata-based citations | Complete |
| Chunking strategy | Complete |
| ChromaDB vector storage | Complete |
| Hugging Face LLM answer generation | Complete |
| Negative-question handling | Complete |
| Retrieval evaluation | Complete |
| Streamlit UI | Complete |
| Docker support | Complete |
| Hugging Face Spaces deployment | Complete |
| Hard benchmark experiment | Complete |
| Reranking experiment | Complete |
| Hybrid retrieval experiment | Complete |
| Prompt versioning | Complete |
| Structured logging | Complete |
| Custom exceptions | Complete |
| LLMOps-lite run tracing | Complete |
| Latency and estimated token tracking | Complete |
| OCR support | Future work |
| RAGAS evaluation | Future work |
| FastAPI backend | Future work |
| Full production LLMOps dashboard | Future work |

---

## Final Note

PolicyDoc AI is built as a practical, evaluated, and publicly deployed RAG project for company policy and compliance documents.

The project demonstrates not only how to build a working RAG pipeline, but also how to evaluate retrieval quality, handle tables, provide citations, test failure cases, add lightweight LLMOps-style observability, and reason about production improvements.


