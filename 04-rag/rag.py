"""A small, visible retrieval-augmented generation (RAG) pipeline.

The model cannot know Northstar Air's private traveler guide. This sample
loads the guide, splits and embeds it, retrieves the chunks closest to a
question, and puts only those chunks in the prompt before the model answers.

The terminal labels every remote API call, local operation, and retrieved
chunk so each part of the RAG flow can be explained while the script runs.

Run:  python rag.py
"""

import os
from pathlib import Path

from databricks.sdk import WorkspaceClient
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
PROFILE = os.environ.get("DATABRICKS_PROFILE")

# These small values make the demo produce several searchable pieces without
# hiding the whole guide in one chunk. Overlap preserves context at boundaries.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 2

# The checked-in file keeps the sample portable. RAG_DOCUMENT may instead be a
# local path or a /Volumes/... path when demonstrating governed source data.
LOCAL_GUIDE = Path(__file__).with_name("traveler-guide.md")
DOCUMENT_PATH = os.environ.get("RAG_DOCUMENT", str(LOCAL_GUIDE))
QUESTION = os.environ.get(
    "RAG_QUESTION",
    "How early should I arrive for an international flight, and when do doors close?",
)


def format_context(documents: list[Document]) -> str:
    """Turn retrieved chunks into the labelled context sent to the chat model.

    Args:
        documents: Chunks in retrieval-rank order, best match first.

    Returns:
        One string containing chunks labelled ``[1]``, ``[2]``, and so on.
        The model uses these same labels as citations in its answer.
    """
    return "\n\n".join(
        f"[{number}] {document.page_content}" for number, document in enumerate(documents, 1)
    )


def load_guide(workspace: WorkspaceClient) -> Document:
    """Load the guide from disk or, optionally, a Unity Catalog Volume.

    A local file is read entirely inside Python. A ``/Volumes/...`` path makes
    one Databricks file-download request. This function only loads text; it
    does not split, embed, retrieve, or call the chat model.

    Args:
        workspace: Authenticated Databricks client used only for Volume files.

    Returns:
        A LangChain document containing the guide text and its source path.
    """
    if DOCUMENT_PATH.startswith("/Volumes/"):
        # REMOTE only when RAG_DOCUMENT points at Unity Catalog.
        download = workspace.files.download(DOCUMENT_PATH)
        if download.contents is None:
            raise RuntimeError(f"Databricks returned no content for {DOCUMENT_PATH}")
        with download.contents as stream:
            text = stream.read().decode("utf-8")
    else:
        text = Path(DOCUMENT_PATH).read_text()

    return Document(page_content=text, metadata={"source": DOCUMENT_PATH})


def main() -> None:
    """Run INDEX -> RETRIEVE -> GENERATE and print every boundary.

    INDEX embeds document chunks remotely and stores their vectors locally.
    RETRIEVE embeds the question remotely, then searches those vectors locally.
    GENERATE sends only the question and retrieved chunks to the chat model.
    """
    # Authenticate once, then share this short-lived OAuth token between the
    # embedding and chat clients. No API call happens merely by creating them.
    workspace = WorkspaceClient(profile=PROFILE)
    token = workspace.config.authenticate()["Authorization"].removeprefix("Bearer ")
    base_url = f"{workspace.config.host}/serving-endpoints"

    # STEP 1 — INDEX: load -> split -> embed -> store.
    # Loading and splitting are local by default. A real source might instead
    # be a collection of private manuals, wiki pages, or support articles.
    document = load_guide(workspace)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents([document])

    # Configuring the embedding client does not make a request. The first
    # embedding API call happens in from_documents() immediately below.
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=token,
        base_url=base_url,
        check_embedding_ctx_length=False,
    )
    print("=== 1. INDEX ===")
    print(f"[API CALL] Embedding {len(chunks)} document chunks with {EMBEDDING_MODEL}...")
    # REMOTE: embed all chunks. LOCAL: retain returned vectors and source text.
    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
    print(f"[LOCAL] Stored {len(chunks)} vectors in memory.\n")

    # STEP 2 — RETRIEVE: embed the question -> compare vectors -> return text.
    print("=== 2. RETRIEVE ===")
    print(f"question> {QUESTION}\n")
    print(f"[API CALL] Embedding the question with {EMBEDDING_MODEL}...")
    # REMOTE: turn the question into a vector using the same embedding model.
    question_vector = embeddings.embed_query(QUESTION)

    # LOCAL: cosine similarity compares the question vector with every stored
    # chunk vector. There is no Vector Search or AI Search request here.
    print("[LOCAL] Comparing the question vector with the in-memory chunk vectors...")
    matches = vector_store.similarity_search_with_score_by_vector(question_vector, k=TOP_K)

    # Keep the ranking order. These exact texts become the model's evidence.
    retrieved = [document for document, _score in matches]
    context = format_context(retrieved)

    # Print the evidence before generation so it is obvious what the model saw.
    # Scores rank chunks for this question; they are not answer confidence.
    print(f"[RETRIEVED] {len(matches)} chunks:\n")
    for number, (document, score) in enumerate(matches, 1):
        print(f"[{number}] similarity={score:.3f}")
        print(f"{document.page_content.strip()}\n")

    # STEP 3 — GENERATE: question + retrieved evidence -> cited answer.
    # Configuring this client is local; model.invoke() is the chat API call.
    model = ChatOpenAI(
        model=MODEL,
        api_key=token,
        base_url=base_url,
        max_tokens=300,
    )
    print("=== 3. GENERATE ===")
    print(f"[API CALL] Sending the question and {len(retrieved)} chunks to {MODEL}...")
    # REMOTE: the system message contains only retrieved evidence, not the
    # entire guide. The grounding instruction also handles missing answers.
    response = model.invoke(
        [
            SystemMessage(
                "Answer only from the retrieved traveler-guide excerpts. "
                "If they do not contain the answer, say you do not know. "
                "Cite supporting excerpts like [1] or [2].\n\n"
                f"Retrieved excerpts:\n{context}"
            ),
            HumanMessage(QUESTION),
        ]
    )

    print(f"answer> {response.content}")


if __name__ == "__main__":
    main()
