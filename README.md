# mrag

A from-scratch C++17 retrieval-augmented generation engine built directly on
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s C API.

No LangChain, LlamaIndex, FAISS, or SQLite. Chunking, UTF-8 handling, embeddings, the vector
store, the binary on-disk format, prompt construction, and the autoregressive sampling loop are
all implemented in this repository.

> **Status: work in progress.** The ingest/chat pipeline runs end to end. Evaluation tooling and
> stronger retrieval are the current focus — see [ROADMAP.md](ROADMAP.md).

---

## What it does

Two modes:

| Mode | Description |
|---|---|
| `--ingest` | Read a text file → split into chunks → embed each chunk → persist a binary vector database |
| `--chat` | Load the database → embed the question → retrieve top-k chunks → stream a grounded answer |

Everything runs locally. No network access at inference time.

---

## Features

- **Section-aware chunking** — cuts on sentence-ending punctuation where possible, always aligned
  to UTF-8 character boundaries (never splits a multi-byte character).
- **Chunk overlap** — configurable byte overlap between adjacent chunks.
- **Binary vector store** — length-prefixed records with a magic number and version header, so
  incompatible files are rejected on load instead of producing garbage.
- **Top-k cosine retrieval** — bounded min-heap, `O(n log k)` rather than sorting the whole corpus.
- **Streaming generation** — tokens are printed as they are sampled; generation stops on EOG.
- **ChatML prompts** — retrieved chunks are injected with source labels.
- **RAII resource management** — model and context handles are owned by a base class with deleted
  copy operations, so double-free is impossible by construction.
- **Optional CUDA build** — `-DMRAG_CUDA=ON`.

---

## Architecture

```
main.cc
  └── MragApp ──┬── buildKnowledgeBase()
                │     DocumentProcessor → EmbeddingEngine → VectorDatabase
                └── chatLoop()
                      VectorDatabase.search() → GenerationEngine

EmbeddingEngine ─┐
                 ├── inherit ── LlamaModelBase (owns llama_model* / llama_context*)
GenerationEngine ┘
```

| File | Responsibility |
|---|---|
| `src/main.cc` | Entry point, CLI parsing, top-level error handling |
| `src/mragapp.*` | Orchestrates ingest and chat; `BackendGuard` scopes backend init/free |
| `src/config.*` | Loads `config.json`; missing fields fall back to defaults |
| `src/document.*` | Reads text, detects section titles, chunks with UTF-8 alignment |
| `src/chunk.*` | Data record + binary serialization |
| `src/vectordatabase.*` | Insert, cosine top-k search, persistence |
| `src/llamamodel.*` | `LlamaModelBase`: RAII wrapper, tokenization |
| `src/embeddingengine.*` | Text → embedding vector |
| `src/generationengine.*` | Prompt assembly, prefill, sampler chain, streaming decode |

See [DESIGN.md](DESIGN.md) for design rationale and known limitations.

---

## Build

### Requirements

- Linux x86-64
- A C++17 compiler (g++ or clang++)
- CMake ≥ 3.16
- Git (dependencies are fetched at configure time)
- Optional: CUDA Toolkit ≥ 12.8 for a GPU build

```bash
sudo apt install -y build-essential cmake git
```

`llama.cpp` (tag `b11528`) and `nlohmann/json` (tag `v3.11.3`) are downloaded automatically by
`FetchContent` into the build directory. **The first configure therefore needs network access.**

### CPU build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### CUDA build

```bash
cmake -S . -B build -DMRAG_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

For RTX 50-series (Blackwell, `sm_120`) add `-DCMAKE_CUDA_ARCHITECTURES=120` and make sure the
CUDA Toolkit is 12.8 or newer — older toolkits cannot compile for `sm_120`.

> After building, confirm the GPU is actually used: llama.cpp prints the offload devices at
> startup. If no GPU device line appears, the build silently fell back to CPU — check
> `GGML_CUDA` in `build/CMakeCache.txt`.

The binary is produced at `build/mrag`.

---

## Models

Two GGUF files are required. Put them in `models/` (paths are read from `config.json`):

| Purpose | Suggested file | Approx. size |
|---|---|---|
| Embedding | `bge-small-zh-v1.5-f16.gguf` | 48 MB |
| Generation | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1.1 GB |

These correspond to the Hugging Face repositories `BAAI/bge-small-zh-v1.5` and
`Qwen/Qwen2.5-1.5B-Instruct`. To obtain GGUF versions, either download a published GGUF
conversion or convert the original weights with llama.cpp's `convert_hf_to_gguf.py`.

Any embedding model and any instruction-tuned generation model supported by llama.cpp can be
used — just point `config.json` at them. Note that changing the embedding model changes the
vector dimension, so existing databases must be rebuilt.

---

## Usage

Run from the repository root; paths in `config.json` and on the command line are resolved
relative to the current directory.

### Build a knowledge base

```bash
./build/mrag --ingest examples/sample.txt sample.db
```

Progress is printed as `embedded i/N`.

### Ask questions

```bash
./build/mrag --chat sample.db
```

```
Loaded knowledge base: sample.db  (4 chunks)
Type a question, or 'exit' / 'quit' to leave.

> Where did Lin Wan find the letter?
A: Lin Wan found the letter in the corner of the old bookshop.

> exit
Bye.
```

Answers are grounded in the retrieved chunks. If the retrieved context does not contain the
answer, the model is instructed to say so rather than invent one.

---

## Configuration

All settings live in `config.json`. Missing sections or fields fall back to built-in defaults,
and a missing file is not an error.

| Section | Field | Meaning | Default |
|---|---|---|---|
| `models` | `embedding` / `generation` | GGUF paths | see `config.json` |
| | `n_gpu_layers` | Layers offloaded to GPU (0 = CPU only, 99 = all) | 0 |
| `document` | `chunk_size` | Target chunk size in bytes | 500 |
| | `overlap_size` | Overlap between adjacent chunks, in bytes | 50 |
| `retrieval` | `top_k` | Number of chunks passed to the generator | 3 |
| `generation` | `n_ctx` / `n_batch` / `n_ubatch` | Generation context parameters | 4096 / 4096 / 512 |
| | `max_output_tokens` | Maximum tokens per answer | 512 |
| | `temperature` / `top_k` / `top_p` | Sampler chain parameters | 0.7 / 40 / 0.9 |
| `embedding` | `n_ctx` / `n_batch` / `n_ubatch` | Embedding context parameters | 512 / 512 / 512 |

---

## Limitations

- Retrieval is exhaustive: every query scans the whole corpus. Fine up to roughly 10⁵ chunks;
  an ANN index is needed beyond that.
- Chunking is byte-based, not semantic. Sentence-boundary alignment reduces but does not
  eliminate mid-sentence splits.
- Section detection matches `第…章/节/回/卷` and `Chapter N` prefixes only.
- No multi-turn conversation state.
- One document per `--ingest` run.

---

## License

MIT — see [LICENSE](LICENSE).
