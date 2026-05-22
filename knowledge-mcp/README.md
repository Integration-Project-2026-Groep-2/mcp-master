# knowledge-mcp

Read-only MCP server exposing one tool, `search_docs`: **hybrid retrieval**
(BM25 + dense embeddings, fused with Reciprocal Rank Fusion) over a curated
corpus of project docs. It lets the Jarvis master agent answer conceptual
questions ("how does Contract 7 work?") grounded in the docs, with source
citations.

It is a standalone MCP server. The master agent (mcp-master) consumes it as
just another entry in `MCP_SERVERS` — no master-agent code change required.
Because the agent calls `search_docs` inside its tool-loop, this is **agentic
RAG**: the LLM decides whether/what to retrieve.

## How it works

1. **Ingest (at startup):** load every `.md` under `CORPUS_DIR`, chunk it
   (structure-aware markdown splitting), embed each chunk with a local ONNX
   model (`multilingual-e5-small`, NL+EN, no API key), and build an in-memory
   BM25 index.
2. **Query (per `search_docs` call):** embed the query, run dense cosine + BM25
   in parallel, fuse the two rankings with RRF, return the top chunks as text
   with their source file.

BM25 is essential for ID-heavy lookups ("Contract 7", XSD element names) where
pure dense vectors are weak; dense covers conceptual / synonym / cross-language
questions. No vector DB — in-memory is correct at this corpus scale.

## Run

```bash
CORPUS_DIR=/path/to/docs PORT=7099 cargo run --release
```

| Env var | Default | Purpose |
|---|---|---|
| `CORPUS_DIR` | `corpus` | Directory of `.md` docs to index (read recursively) |
| `PORT` | `7099` | HTTP port; serves Streamable-HTTP MCP at `/mcp` |

First run downloads the e5 ONNX model (~120 MB) into `.fastembed_cache/`;
cache it in the container image for fast boots. `/mcp` is unauthenticated and
meant for in-cluster scraping/use only — do not expose it publicly.

## Register with mcp-master

Append it to the master agent's `MCP_SERVERS` (alongside crm / controlroom):

```
MCP_SERVERS=crm@http://crm:7001/mcp,controlroom@http://controlroom:5555/mcp,knowledge@http://knowledge-mcp:7099/mcp
```

`search_docs` declares `readOnlyHint=true`, so mcp-master routes it read-only
(no approval flow). Verify with `mcp-master --list-tools`.
