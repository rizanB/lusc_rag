import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import numpy as np
import shutil
from typing import List

st.set_page_config(page_title="lusc_rag | localChatbot", page_icon="🧬")


# configure local models
@st.cache_resource
def load_local_models():
    try:
        llm = Ollama(model="llama3.2:3b")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return llm, embeddings, "ollama"
    except:
        # fallback to cpu-only SentenceTransformers
        st.warning("Ollama not found. Using SentenceTransformers")
        llm = None
        embeddings = SentenceTransformerEmbeddings()
        return llm, embeddings, "sentence_transformers"


class SentenceTransformerEmbeddings:
    """custom embeddings"""

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()


def simple_llm_response(context: str, question: str) -> str:
    """basic fallback"""
    # return basic context only
    return f"""Based on papers, here's what I found:

{context[:2000]}...

(Note: Install and configure to use Ollama for better responses)
"""


# read local files
def fetch_local_content(folder_path="recent_advances"):
    """combine txt files into documents"""
    documents = []

    if not os.path.exists(folder_path):
        st.error(f"Folder '{folder_path}' not found. You sure recent_advances exists?")
        return []

    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    if not txt_files:
        st.warning(f"No text files in '{folder_path}' folder")
        return []

    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

                chunks = [content[i : i + 1000] for i in range(0, len(content), 800)]

                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "chunk": i,
                            "paper_name": filename.replace(".txt", ""),
                        },
                    )
                    documents.append(doc)

                st.success(f"Loaded: {filename} ({len(chunks)} chunks)")
        except Exception as e:
            st.error(f"Error reading {filename}: {str(e)}")

    return documents


st.title("lusc_rag | localChatbot")
st.write("Ask questions about squamous cell cancer research.")

model_option = st.selectbox(
    "Choose your setup:",
    ["Auto-detect", "Force SentenceTransformers (CPU only)", "Force Ollama"],
)

if model_option == "Force SentenceTransformers (CPU only)":
    llm, embeddings, model_type = (
        None,
        SentenceTransformerEmbeddings(),
        "sentence_transformers",
    )
elif model_option == "Force Ollama":
    try:
        llm = Ollama(model="llama3.2:3b")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        model_type = "ollama"
    except Exception as e:
        st.error(f"Ollama failed: {e}")
        st.stop()
else:
    llm, embeddings, model_type = load_local_models()

st.info(f"Using: {model_type}")

if st.button("Load/Reload Documents"):
    with st.spinner("loading documents..."):
        documents = fetch_local_content()

        if documents:

            # embeddings and vec store
            with st.spinner("Creating embeddings..."):

                if model_type == "ollama":
                    persist_dir = "chroma_store_ollama"
                else:
                    persist_dir = "chroma_store_sentence_transformers"

                # clear existing store when switching betn models
                import shutil

                if os.path.exists(persist_dir):
                    st.info(f"Clearing existing {model_type} database...")
                    shutil.rmtree(persist_dir)

                vector_store = Chroma.from_documents(
                    documents, embeddings, persist_directory=persist_dir
                )
                vector_store.persist()
                st.session_state.vector_store = vector_store
                st.session_state.current_model = model_type
                st.success(
                    f"Loaded {len(documents)} document chunks using {model_type}!"
                )
        else:
            st.error("No documents loaded. Please check 'recent_advances' folder.")

# init vec store
if "vector_store" not in st.session_state:
    st.info("👆 Click 'Load/Reload Documents' to start!")
    st.stop()

# check if reload needed
current_model = getattr(st.session_state, "current_model", None)
if current_model and current_model != model_type:
    st.warning(
        f"Model changed from {current_model} to {model_type}. Please reload documents!"
    )
    st.stop()

# init session state, for convo history
if "messages" not in st.session_state:
    st.session_state.messages = []

# convo history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# interface for input
if prompt := st.chat_input("Ask about lung cancer research:"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # rag response
    if "vector_store" in st.session_state:
        docs = st.session_state.vector_store.similarity_search(prompt, k=5)

        # combine content from documents
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            chunk = doc.metadata.get("chunk", 0)
            content = doc.page_content
            context_parts.append(f"From {source} (chunk {chunk}):\n{content}")

        context = "\n\n---\n\n".join(context_parts)

        with st.spinner("Generating response..."):
            if llm and model_type == "ollama":

                question_template = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""You are an expert in lung cancer research. Based on the following research paper excerpts, provide accurate answer to the question.

Context from research papers:
{context}

Question: {question}

Please provide a brief and accurate answer based on the research papers, and mention which papers you're referencing when possible.""",
                )

                formatted_prompt = question_template.format(
                    context=context, question=prompt
                )
                answer_content = llm(formatted_prompt)
            else:
                # simple fallback
                answer_content = simple_llm_response(context, prompt)

        with st.chat_message("assistant"):
            st.markdown(answer_content)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer_content}
        )

        # save convo history
        with open("conversation_log.txt", "a", encoding="utf-8") as f:
            f.write(f"User: {prompt}\nChatbot: {answer_content}\n\n")
    else:
        st.error("Please load documents first!")

# instructions in sidebar
with st.sidebar:
    st.markdown("## Instructions")
    st.markdown(
        """
    **SentenceTransformers (cpu-only): Text-search only, basic, but faster**
    ```bash
    pip install sentence-transformers
    ```
    
    **Ollama (gpu-accelerated) : Better, but slower**
    ```bash
    # Install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh1
    
    # Pull models
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text
    ```
    
    Recommended: Try Ollama first! If it's slower, try the llama3.2:1b or reduce document chunks in search.
    """
    )
