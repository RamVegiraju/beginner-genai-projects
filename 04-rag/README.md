# 4 — Retrieval-augmented generation (RAG)

The weather agent in sample 3 can call an API for facts that change. It still
cannot know a private traveler guide that was never part of its training data.
This sample retrieves the relevant parts of that guide before asking the model
to answer.

**RAG in one sentence:** search your documents for useful evidence, then give
that evidence to the model with the user's question.

## Basic RAG architecture

RAG has two phases. **Indexing** prepares the documents. **Retrieval and
generation** happen when a user asks a question.

```text
INDEXING

traveler-guide.md
        │
        ▼
load document → split into chunks → embedding API call → local vector store


QUESTION TIME

user question → embedding API call → local similarity search (no API call)
                                               │
                                               ▼
                                      2 most relevant chunks
                                               │
                                               ▼
                               question + chunks → chat model → cited answer
```

The parts in beginner language:

| Part | What it does | This sample uses |
|---|---|---|
| Source document | Contains facts the model needs | `traveler-guide.md` |
| Loader | Reads the document into Python | `Path.read_text()` or a Unity Catalog Volume download |
| Text splitter | Breaks a long document into searchable pieces | `RecursiveCharacterTextSplitter` |
| Embedding model | Turns text into lists of numbers that represent meaning | `databricks-gte-large-en` |
| Vector store | Keeps the chunks and their vectors | LangChain `InMemoryVectorStore` |
| Retriever | Finds chunks whose meaning is closest to the question | Local similarity search with `k=2` |
| Chat model | Writes the final answer from the retrieved evidence | The configured Databricks chat endpoint |

The embedding model does **not** write the answer. It converts both document
chunks and the question into vectors so similar meanings can be matched. The
chat model receives the question plus the two retrieved chunks and writes the
answer.

All vector storage and similarity search happen inside the local Python
process. This sample does not create or use an AI Search or Databricks Vector
Search endpoint.

## What makes an API call?

Retrieval is easy to misunderstand because embedding and search are often
wrapped in one method. This sample keeps them separate and prints each step:

| Step | API call? | What happens |
|---|---|---|
| Load the checked-in guide | No | Python reads the Markdown file. A Unity Catalog source makes one file-download request instead. |
| Split the guide | No | Python creates overlapping text chunks. |
| Embed document chunks | **Yes** | The chunks are sent to `databricks-gte-large-en`. |
| Store vectors | No | `InMemoryVectorStore` keeps them in local memory. |
| Embed the question | **Yes** | The question is sent to the same embedding model. |
| Retrieve two chunks | No | Python compares the question vector with the stored vectors locally. |
| Generate the answer | **Yes** | The question and retrieved text are sent to the chat model. |

The terminal prints `[API CALL]`, `[LOCAL]`, and `[RETRIEVED]` labels as it
runs. It also prints the exact retrieved chunk text and its similarity score
before making the final chat-model call.

## What the sample contains

The fictional Northstar Air guide covers:

- carry-on and checked bags;
- airport arrival and boarding times;
- ticket changes and cancellations;
- delays and severe weather; and
- airport assistance.

The script loads that guide, creates five overlapping chunks, embeds them,
retrieves the two closest chunks, and asks the chat model to answer only from
that evidence. It also asks the model to cite the retrieved chunks as `[1]` or
`[2]` and to say it does not know when the evidence is missing.

## Setup

### 1. Complete the shared setup

Follow [SETUP.md](../SETUP.md) once to install the Databricks CLI, authenticate,
and create the Python 3.12 virtual environment.

In each new terminal, select the profile you created:

```bash
export DATABRICKS_PROFILE=genai-series
```

The workspace needs one ready chat endpoint and one ready embedding endpoint.
The defaults are:

```text
Chat:       databricks-claude-haiku-4-5
Embeddings: databricks-gte-large-en
```

If either default is unavailable, list the endpoints in your workspace and use
ready endpoints from that list:

```bash
databricks serving-endpoints list --profile genai-series

export SERVING_ENDPOINT=<a-ready-chat-endpoint>
export EMBEDDING_ENDPOINT=<a-ready-embedding-endpoint>
```

### 2. Install this sample

Run this from the repository root:

```bash
uv pip install --python .venv/bin/python -r 04-rag/requirements.txt
```

No database, search service, or additional API key is required.

## Run the default example

From the repository root:

```bash
cd 04-rag
../.venv/bin/python rag.py
```

The default question is:

```text
How early should I arrive for an international flight, and when do doors close?
```

Measured output on September 22, 2026:

```text
=== 1. INDEX ===
[API CALL] Embedding 5 document chunks with databricks-gte-large-en...
[LOCAL] Stored 5 vectors in memory.

=== 2. RETRIEVE ===
question> How early should I arrive for an international flight, and when do doors close?

[API CALL] Embedding the question with databricks-gte-large-en...
[LOCAL] Comparing the question vector with the in-memory chunk vectors...
[RETRIEVED] 2 chunks:

[1] similarity=0.846
## Arriving and boarding

Arrive at the airport at least two hours before a domestic flight and three
hours before an international flight. Boarding begins 40 minutes before
departure. The boarding door closes 15 minutes before departure, even if the
aircraft is still at the gate.

## Changes and cancellations

[2] similarity=0.612
## Help at the airport

Travelers who need wheelchair assistance should request it at least 48 hours
before departure. Unaccompanied minors and travelers with pets must check in
with an airport agent; they cannot complete check-in only through the app.

=== 3. GENERATE ===
[API CALL] Sending the question and 2 chunks to databricks-gpt-5-4-mini...
answer> You should arrive at the airport at least three hours before an
international flight. The boarding door closes 15 minutes before departure,
even if the aircraft is still at the gate. [1]
```

The exact wording can vary, but the answer should contain the same grounded
facts and a citation. Similarity scores rank chunks for this question; they are
not probabilities or measures of whether the final answer is correct. Notice
that chunk `[1]` contains the answer while chunk `[2]` is less relevant.

## Ask your own questions

Use `RAG_QUESTION` so you do not need to edit the Python file:

```bash
RAG_QUESTION="What is the change fee for a standard fare?" \
  ../.venv/bin/python rag.py
```

More requests to try:

| Request | What you should observe |
|---|---|
| `How many bags can I bring, and how large can my carry-on be?` | Retrieves the baggage section and cites it. |
| `What happens if Northstar cancels my flight because of weather?` | Retrieves the delay and rebooking policy. |
| `When can I get a full refund after booking?` | Retrieves the cancellation rules. |
| `Does the airport lounge serve breakfast?` | Says it does not know because the guide never mentions a lounge. |

For example:

```bash
RAG_QUESTION="Does the airport lounge serve breakfast?" \
  ../.venv/bin/python rag.py
```

The last request is important. RAG can retrieve facts that exist in the source;
it cannot recover information that was never indexed.

## Optional: load the guide from Unity Catalog

This is not required. The checked-in guide is the default so the sample stays
portable and easy to run.

If you already use Unity Catalog, you can keep the source document in an
existing Volume. From the repository root, upload it once:

```bash
databricks fs cp 04-rag/traveler-guide.md \
  dbfs:/Volumes/<catalog>/<schema>/<volume>/northstar-air-traveler-guide.md \
  --profile genai-series
```

Then point the sample at it:

```bash
export RAG_DOCUMENT=/Volumes/<catalog>/<schema>/<volume>/northstar-air-traveler-guide.md

cd 04-rag
../.venv/bin/python rag.py
```

Only the source file lives in the Volume. The script downloads the text and
still performs chunking, embedding storage, and similarity search locally. It
does not create a table, vector index, or search endpoint.

## How the code maps to the architecture

Open [`rag.py`](rag.py) and follow these sections in order:

1. `load_guide()` reads the local or Unity Catalog document.
2. `RecursiveCharacterTextSplitter` creates 500-character chunks with overlap.
3. `OpenAIEmbeddings` calls the Databricks GTE embedding endpoint.
4. `InMemoryVectorStore.from_documents(...)` builds the local index.
5. `embed_query(...)` makes the question-embedding API call.
6. `similarity_search_with_score_by_vector(..., k=2)` searches locally.
7. The retrieved chunks and similarity scores are printed.
8. `ChatOpenAI.invoke(...)` generates an answer from that evidence.

The same embedding model is used for the document chunks and question. Mixing
embedding models would make their vectors incompatible.

## Why this is not another agent

Every question in this sample should search the same guide, so there is no
decision for an agent to make. A direct RAG pipeline is smaller and easier to
understand. If a future application needs to choose between this guide, live
weather, and other tools, the retriever can become a tool in the LangChain
agent from sample 3.

## Deliberate limits

- The vector store disappears when the script exits.
- The index is rebuilt on every run.
- One small Markdown file stands in for a document collection.
- The labels identify retrieved chunks, not page-level citations.

These choices keep the entire example readable in one sitting. A production
system may add persistent storage, incremental indexing, access control, and
retrieval evaluation when its scale requires them.

## References

- [LangChain semantic search tutorial](https://docs.langchain.com/oss/python/langchain/knowledge-base)
- [Databricks Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/)

## Next

RAG gives the model knowledge outside its training data. It still forgets the
conversation when the process ends. [Sample 5](../05-agent-memory/) adds
short-term and long-term memory.
