"""Responsive Streamlit UI for persistent, collection-based RAG."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

import rag_engine
from chat_store import ChatStore
from collection_manager import CollectionError, CollectionManager
from utils import get_logger, load_config


config = load_config()
logger = get_logger(__name__)
manager = CollectionManager()
chat_store = ChatStore()

st.set_page_config(page_title="Knowledge Workspace", page_icon="🧠", layout="wide")


def render_sources(sources: list, expanded: bool = False) -> None:
    if not sources:
        return
    with st.expander("Sources", expanded=expanded):
        for number, source in enumerate(sources, 1):
            st.markdown(
                f"**[{number}]** **{source.get('source_collection', 'legacy')}** · "
                f"`{source['source_file']}` · chunk **{source['chunk_id']}** · "
                f"score **{source['score']:.4f}**"
            )
            excerpt = source.get("text", "")[:400]
            st.caption(excerpt + ("…" if len(source.get("text", "")) > 400 else ""))


def ensure_active_chat(knowledge_bases: list[str]) -> str:
    chats = {chat["id"] for chat in chat_store.list()}
    current = st.session_state.get("active_chat_id")
    if current not in chats:
        current = chat_store.create(knowledge_bases)["id"]
        st.session_state.active_chat_id = current
    return current


manager.discover_collections()
knowledge_bases = manager.list_collections()
active_chat_id = ensure_active_chat([])
active_chat = chat_store.get(active_chat_id)
if st.session_state.get("scope_chat_id") != active_chat_id:
    st.session_state.scope_chat_id = active_chat_id
    st.session_state.search_bases = [
        base for base in active_chat["knowledge_bases"] if base in knowledge_bases
    ]

st.title("🧠 Knowledge Workspace")
st.caption("Persistent knowledge bases, grounded answers, and durable conversation history.")

with st.sidebar:
    st.header("Knowledge bases")
    with st.form("create_knowledge_base", clear_on_submit=True):
        new_name = st.text_input("New knowledge base", placeholder="client-contracts")
        create_clicked = st.form_submit_button("Create knowledge base")
    if create_clicked:
        try:
            if manager.create_collection(new_name):
                st.success(f"Created {new_name}.")
                st.rerun()
            st.info("That knowledge base already exists.")
        except CollectionError as error:
            st.error(str(error))

    active_base = st.selectbox(
        "Open knowledge base", knowledge_bases, index=None,
        placeholder="Select a knowledge base to manage documents",
        disabled=not knowledge_bases,
    ) if knowledge_bases else None

    scope_left, scope_right = st.columns(2)
    if scope_left.button("All", use_container_width=True):
        st.session_state.search_bases = knowledge_bases
    if scope_right.button("Clear", use_container_width=True):
        st.session_state.search_bases = []
    selected_bases = st.multiselect(
        "Search scope",
        knowledge_bases,
        key="search_bases",
        help="Select one, several, or All. This affects answers; Open knowledge base only manages files.",
    )
    if selected_bases != active_chat["knowledge_bases"]:
        chat_store.update_knowledge_bases(active_chat_id, selected_bases)
        active_chat = chat_store.get(active_chat_id)

    st.divider()
    st.caption(f"LLM: `{config['models']['llm_model_name']}`")
    st.caption(f"Embeddings: `{config['models']['embedding_model_name']}`")
    score_threshold = st.slider(
        "Similarity threshold", 0.0, 1.0,
        float(config["pipeline"]["relevance_threshold"]), 0.01,
    )
    top_k = st.slider(
        "Retrieved candidates", 1, 20,
        int(config["pipeline"].get("retrieval_candidates", 5)), 1,
        help="How many semantic candidates are ranked before the answer is generated.",
    )
    show_sources = st.checkbox("Show sources", value=True)


if active_base:
    documents = manager.sync_collection(active_base)
    st.subheader(f"Documents · {active_base}")
    pending_count = sum(document["status"] == "pending" for document in documents)

    upload_key = f"upload_{active_base}_{st.session_state.get('upload_nonce', 0)}"
    uploaded_file = st.file_uploader(
        "Upload a document", type=["pdf", "txt", "docx"], key=upload_key,
        help="The original is saved as Pending. Processing starts only when requested.",
    )
    if uploaded_file is not None and st.button("Add as Pending", type="primary"):
        try:
            manager.upload_bytes(active_base, uploaded_file.name, uploaded_file.getvalue())
            st.session_state.upload_nonce = st.session_state.get("upload_nonce", 0) + 1
            st.success(f"{uploaded_file.name} is Pending.")
            st.rerun()
        except CollectionError as error:
            st.error(str(error))

    if pending_count:
        if st.button(f"Process Pending Files ({pending_count})"):
            with st.spinner("Extracting, chunking, embedding, and indexing…"):
                try:
                    summary = manager.process_pending(active_base)
                    st.success(f"Processed {summary['processed']}; {summary['failed']} remain Pending.")
                    st.rerun()
                except Exception as error:
                    logger.exception("Collection processing failed")
                    st.error(f"Processing could not start: {error}")

    if documents:
        st.dataframe([
            {"File": document["filename"], "Status": document["status"],
             "Chunks": document["chunk_count"], "Last error": document["last_error"] or ""}
            for document in documents
        ], use_container_width=True, hide_index=True)
        removable = st.multiselect("Documents to remove", [document["filename"] for document in documents])
        if removable:
            confirmed = st.checkbox(
                "I understand that selected documents will be removed from search and archived locally."
            )
            if st.button("Remove selected documents", disabled=not confirmed):
                try:
                    removed = manager.remove_documents(active_base, removable)
                    st.success(f"Removed {removed} document(s). A recoverable backup was created.")
                    st.rerun()
                except Exception as error:
                    logger.exception("Document removal failed")
                    st.error(f"Documents were not removed: {error}")
else:
    st.info("Create a knowledge base to add documents.")

search_health = manager.qdrant_health(selected_bases)
if selected_bases and not search_health["online"]:
    st.error("Search service is unavailable. Start Qdrant, then retry.")
elif search_health["missing"]:
    missing_label = ", ".join(search_health["missing"])
    st.warning(f"Search index is missing for: {missing_label}.")
    if st.button("Repair missing search indexes"):
        queued = sum(manager.mark_for_reindex(name) for name in search_health["missing"])
        st.success(f"Queued {queued} document(s). Select each knowledge base and process Pending Files.")
        st.rerun()


st.divider()
chat_column, history_column = st.columns([3, 1], gap="large")
with chat_column:
    st.subheader(active_chat["title"])
    for message in active_chat["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            elif message.get("is_general"):
                st.info("General LLM answer — not grounded in selected knowledge bases.")
                st.write(message["answer"])
            elif message.get("above_threshold"):
                st.write(message["answer"])
            else:
                st.warning(message.get("answer", ""))
            if message["role"] == "assistant" and show_sources:
                render_sources(message.get("sources", []))

    question = st.chat_input(
        "Ask your selected knowledge bases…", disabled=not selected_bases
    )
    if question:
        user_message = {"role": "user", "content": question}
        chat_store.append(active_chat_id, user_message)
        with st.chat_message("user"):
            st.write(question)

        qdrant_names = search_health["available"]
        with st.chat_message("assistant"):
            with st.spinner("Searching selected knowledge bases…"):
                try:
                    result = rag_engine.answer_with_rag(
                        question=question, score_threshold=score_threshold,
                        collection_names=qdrant_names, top_k=top_k,
                    )
                except Exception as error:
                    logger.exception("RAG error for query %r", question)
                    result = {"answer": f"⚠️ Error: {error}", "sources": [], "above_threshold": False}
            if result["above_threshold"]:
                st.write(result["answer"])
            else:
                st.warning(result["answer"])
                if st.button("Get a general LLM answer", key=f"general_{active_chat_id}_{len(active_chat['messages'])}"):
                    with st.spinner("Generating a general answer…"):
                        general_answer = rag_engine.answer_with_general_llm(question)
                    st.info("General LLM answer — not grounded in selected knowledge bases.")
                    st.write(general_answer)
                    chat_store.append(active_chat_id, {
                        "role": "assistant", "answer": general_answer,
                        "sources": [], "above_threshold": False, "is_general": True,
                    })
            if show_sources:
                render_sources(result["sources"], expanded=True)
        chat_store.append(active_chat_id, {"role": "assistant", **result})

with history_column:
    st.subheader("Chat history")
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.active_chat_id = chat_store.create(selected_bases)["id"]
        st.rerun()
    if st.button("Save backup", use_container_width=True):
        chat_store.create_backup()
        st.success("Backup saved.")
    if active_chat["messages"] and st.button("Clear current chat", use_container_width=True):
        chat_store.clear(active_chat_id)
        st.rerun()
    st.caption("Conversations are autosaved locally after every message.")
    for chat in chat_store.list():
        marker = "● " if chat["id"] == active_chat_id else ""
        if st.button(marker + chat["title"], key=f"open_{chat['id']}", use_container_width=True):
            st.session_state.active_chat_id = chat["id"]
            st.rerun()
        if chat["id"] == active_chat_id and len(chat_store.list()) > 1:
            if st.button("Delete this chat", key=f"delete_{chat['id']}", use_container_width=True):
                chat_store.delete(chat["id"])
                st.session_state.active_chat_id = chat_store.list()[0]["id"]
                st.rerun()
