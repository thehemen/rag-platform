# RAG Platform

A Retrieval-Augmented Generation platform for Markdown and plain text documents.

The project loads local documents, splits them into chunks, creates SentenceTransformers embeddings, stores vectors in Qdrant, retrieves relevant context, and generates answers with the Groq API. It also includes FastAPI endpoints, a Gradio demo, and evaluation scripts for retrieval and answer quality.

## Features

* Load Markdown and plain text documents from local folders
* Split documents into smaller chunks with LlamaIndex
* Create embeddings with SentenceTransformers
* Store and search vectors with Qdrant
* Generate RAG answers with Groq
* Serve the RAG pipeline through FastAPI
* Test the system in a browser with Gradio
* Prepare a small evaluation dataset
* Run retrieval evaluation
* Run answer-quality evaluation
* Clean model outputs before displaying or saving them

## Stack

* Python
* Docker
* FastAPI
* Gradio
* LlamaIndex
* SentenceTransformers
* Qdrant
* Groq API
* DeepEval-compatible evaluation

## Installation

Clone the repository:

```bash
git clone https://github.com/thehemen/rag-platform
cd rag-platform
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Docker Installation

Install Docker if it is not available on your machine yet.

On Ubuntu, one simple option is:

```bash
sudo apt update
sudo apt install -y docker.io
```

Start and enable Docker:

```bash
sudo systemctl start docker
sudo systemctl enable docker
```

Check that Docker works:

```bash
docker --version
```

If you want to run Docker commands without `sudo`, add your user to the Docker group:

```bash
sudo usermod -aG docker $USER
```

Then log out and log in again.

## Environment Variables

Create a `.env` file:

```bash
cp .env.example .env
```

Example configuration:

```env
GROQ_API_KEY=
GROQ_MODEL=qwen/qwen3-32b
GROQ_FALLBACK_MODELS=qwen/qwen3.6-27b

LLM_TEMPERATURE=0.6
LLM_MAX_TOKENS=1024
LLM_TOP_P=0.95
LLM_REASONING_EFFORT=default
LLM_STREAM=false
LLM_MAX_RETRIES=2
LLM_RETRY_SLEEP_SECONDS=2.0
LLM_TIMEOUT=180

EVAL_TEMPERATURE=0.0
EVAL_MAX_TOKENS=512
EVAL_THRESHOLD=0.7

EVAL_GENERATION_MODEL=qwen/qwen3-32b
EVAL_GENERATION_TIMEOUT=180
EVAL_TOP_P=0.95
EVAL_REASONING_EFFORT=default
EVAL_LLM_MAX_RETRIES=1
EVAL_LLM_RETRY_SLEEP_SECONDS=2.0

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=rag_markdown_docs

MARKDOWN_DIR=data/raw/markdown
PLAINTEXT_DIR=data/raw/plaintext

EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=
EMBEDDING_BATCH_SIZE=32

CHUNK_SIZE=512
CHUNK_OVERLAP=80
TOP_K=3
RECREATE_COLLECTION=true
```

## Data Layout

Place Markdown files here:

```text
data/raw/markdown/
```

Place plain text files here:

```text
data/raw/plaintext/
```

Example:

```text
data/raw/
├── markdown/
│   ├── kitti-yolo-detector.md
│   └── minetest-assistant.md
└── plaintext/
    └── nfl_impact_detection.txt
```

## Run Qdrant

Start Qdrant with Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
  qdrant/qdrant
```

Qdrant will be available at:

```text
http://localhost:6333
```

## Rebuild the Index

After adding documents, rebuild the vector index:

```bash
python scripts/rebuild_index.py
```

This step loads documents, chunks them, creates embeddings, and stores vectors in Qdrant.

## Test Groq

Run a simple Groq API test:

```bash
python scripts/groq_test.py "Explain RAG in one paragraph."
```

## Ask from CLI

Run a RAG query from the command line:

```bash
python scripts/rag_query.py "What is this project?"
```

Or:

```bash
python scripts/ask_cli.py "What is the NFL impact detection project?"
```

## Run FastAPI

Start the API server:

```bash
python scripts/run_api.py
```

Health check:

```bash
curl http://localhost:8000/health
```

Retrieval-only request:

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this project?", "top_k": 3}'
```

Full RAG request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this project?", "top_k": 3}'
```

## Run Gradio Demo

Start FastAPI first:

```bash
python scripts/run_api.py
```

Then start Gradio in another terminal:

```bash
python scripts/run_gradio.py
```

Open:

```text
http://localhost:7860
```

The Gradio demo lets you enter a question, adjust retrieval and generation settings, and inspect the answer, sources, and raw API response.

## Evaluation Data

Prepare a small evaluation split:

```bash
python scripts/prepare_eval_data.py
```

This creates an evaluation dataset with questions, expected answers, and expected source files.

## Retrieval Evaluation

Run retrieval-only evaluation:

```bash
python scripts/run_evals.py --mode retrieval
```

This checks whether the expected source document appears in the retrieved results.

Example output:

```text
Retrieval summary
--------------------------------------------------------------------------------
Examples: 10
Hits: 10
Errors: 0
Hit rate: 1.0000
```

## Answer Evaluation

Run answer-quality evaluation:

```bash
python scripts/run_evals.py --mode deepeval
```

The evaluation checks:

* Answer relevancy
* Faithfulness
* Contextual relevancy
* Correctness

The current implementation uses safe deterministic metrics to avoid failures caused by invalid judge responses.

Reports are written to:

```text
data/eval/reports/retrieval_results.jsonl
data/eval/reports/deepeval_results.jsonl
```

## Current Evaluation Result

The latest retrieval evaluation shows that retrieval works correctly:

```text
Examples: 10
Hits: 10
Errors: 0
Hit rate: 1.0000
```

Answer-quality evaluation is also stable and produces numeric scores without parser failures.

## Notes

This project is designed as a compact RAG platform for portfolio and experimentation purposes. It demonstrates the full workflow from document ingestion to retrieval, answer generation, API serving, browser demo, and evaluation. It's successfully tested on the documents.

Good luck!
