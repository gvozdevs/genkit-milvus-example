# Genkit + Milvus Example (Go)

A minimal example that shows how to build a simple RAG-style workflow in Go using:
- Firebase Genkit for Go (flows, prompts, retrievers)
- Milvus as the vector database
- OpenAI for embeddings and chat completion

The program:
1. Starts a Milvus client and ensures a collection named "products" exists (with a float vector field and text field).
2. Indexes four sample product descriptions using OpenAI embeddings (text-embedding-3-small).
3. Retrieves the top 2 most relevant items for a user query from Milvus.
4. Feeds the retrieved context into a prompt and asks GPT-4o to answer like a phone store salesperson.


## Requirements
- Go 1.25 or newer
- Docker and Docker Compose
- An OpenAI API key (environment variable OPENAI_API_KEY)

Note: This example uses a local replacement for Genkit in `go.mod`:

```
replace github.com/firebase/genkit/go v1.0.2 => /Users/gvo/work/genkit-fork/go
```

If you are not on the same machine or do not have that path, update/remove this replace directive accordingly.


## Getting started

1) Start Milvus locally via Docker Compose (from the project root):

```
docker compose up -d
```

This brings up etcd, MinIO, and a Milvus standalone instance. Milvus will be available at 127.0.0.1:19530 when healthy.

2) Set your OpenAI API key:

macOS/Linux:
```
export OPENAI_API_KEY=your_key_here
```
Windows (PowerShell):
```
$Env:OPENAI_API_KEY = "your_key_here"
```

3) Run the example:

```
go run .
```

On the first run, the app will create and load the `products` collection (if it does not exist), index four tiny sample documents, run a retrieval for the query "I want to buy an iphone", and then call GPT-4o with the retrieved context. You should see a log line like:

```
result: ...
```

4) Stop the services when done:

```
docker compose down
```


## How it works
- Milvus connection: main.go connects to Milvus at 127.0.0.1:19530 (see `milvus.NewEngine(ctx, milvus.WithAddress("127.0.0.1:19530"))`).
- Collection schema (collection name: `products`):
  - Primary key: `id` (int64)
  - Vector: `vector` (FloatVector, dim 1536)
  - Text: `text` (VarChar)
  - An index is created using cosine similarity.
- Embeddings: OpenAI model `text-embedding-3-small` is used to embed documents before indexing.
- Retrieval: Uses Genkit's Milvus retriever, requesting top 2 results.
- Prompt: A simple template instructs the assistant to act as a phone store salesperson and use the retrieved context.


## Configuration
Adjust these in `main.go` if needed:
- Milvus address: `127.0.0.1:19530`
- Collection name: `products`
- Keys and dimensions: `id`, `vector` (dim 1536), `text`
- Embedding model: `text-embedding-3-small`
- Chat model: `openai/gpt-4o`
- Retrieval limit: 2

Also ensure the `go.mod` `replace` directive for Genkit points to a valid path for your environment, or remove it to use the published module.


## Troubleshooting
- Milvus not ready: If the app fails on first connection, wait until containers are healthy (`docker compose ps`) and try again.
- Missing OPENAI_API_KEY: Set the environment variable before running.
- Network or model errors: Ensure your network allows calls to OpenAI and that your account has access to the specified models.
- Schema/compatibility: If you change vector dimensions or model, update both the Milvus schema and the embedder dimension accordingly.
