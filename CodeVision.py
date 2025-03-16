# -------------------------------------------------
# Setup: Poppler and Tesseract Paths
# -------------------------------------------------
import os
poppler_bin_path = r"C:\Program Files (x86)\poppler-24.08.0\Library\bin"
tesseract_bin_path = r"C:\Program Files\Tesseract-OCR"
os.environ["PATH"] = poppler_bin_path + os.pathsep + tesseract_bin_path + os.pathsep + os.environ["PATH"]


import re
import io
import base64
import json
import shutil
import numpy as np
from typing import List, Dict, Any, TypedDict, Optional
import json

import torch
torch.classes.__path__ = []

from PIL import Image

import chromadb
import ollama
from transformers import CLIPProcessor, CLIPModel

# LangChain and related imports
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.prompts import PromptTemplate

# LangGraph
from langgraph.graph import StateGraph, START, END

# LangChain Agents
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage

import streamlit as st

# Unstructured functions
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.basic import chunk_elements

# -------------------------------------------------
# 1. Helper Functions
# -------------------------------------------------
def clean_metadata(metadata):
    if not isinstance(metadata, dict):
        return {}
    out = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)) and v is not None:
            out[k] = v
        elif isinstance(v, (list, dict)):
            out[k] = json.dumps(v)
    return out

def get_text_embedding_for_query(query_text: str) -> torch.Tensor:
    response = ollama.embeddings(model="mxbai-embed-large", prompt=query_text)
    embedding = response["embedding"]
    return torch.tensor(embedding)

def embed_documents(docs: List[str]) -> List[List[float]]:
    return [get_text_embedding_for_query(doc).tolist() for doc in docs]

# -------------------------------------------------
# 2. RAG_embedding Class
# -------------------------------------------------
class RAG_embedding:
    
    def __init__(self, folder_path: str, prefix: str, suffix: str, db_path: str = "chroma_db", use_unstructured: bool = True):
        self.folder_path = folder_path
        self.prefix = prefix
        self.suffix = suffix
        self.db_path = db_path
        self.use_unstructured = use_unstructured

        self.elements = []  # Text elements
        self.image_metadata: Dict[str, List[Dict[str, str]]] = {}  # { page_key: [{ "path": ..., "caption": ..., "base64": ... }, ...] }

        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.vision_llm = Ollama(model="llama3.2-vision", temperature=0.0, format=None)

    def extract_content(self):
        base64_images = []
        pdf_files = [f for f in os.listdir(self.folder_path) if f.startswith(self.prefix) or f.endswith(self.suffix)]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.folder_path, pdf_file)
            try:
                elems = partition_pdf(
                    filename=pdf_path,
                    strategy="hi_res",
                    languages=["eng"],
                    extract_images_in_pdf=True,
                    extract_image_block_types=["Image", "Table", "Figure"],
                    extract_image_block_to_payload=False  # <-- Turned off payload extraction for the model to save the image for instepction.
                )
            except Exception as e:
                print(f"⚠️ '{pdf_file}': {e}. Falling back to ocr_only strategy.")
                try:
                    elems = partition_pdf(
                        filename=pdf_path,
                        strategy="ocr_only",
                        languages=["eng"],
                        extract_images_in_pdf=True,
                        extract_image_block_types=["Image", "Table", "Figure"],
                        extract_image_block_to_payload=False  # <-- Also here
                    )
                except Exception as e2:
                    print(f"⚠️ Error processing {pdf_file} with fallback: {e2}")
                    continue

            for element in elems:
                page_number = getattr(element.metadata, "page_number", None)
                page_key = f"{pdf_file}_page_{page_number}" if page_number is not None else pdf_file

                if element.category in ["NarrativeText", "Title", "Text", "List", "Table"]:
                    self.elements.append(element)
                elif element.category in ["Image", "Figure"]:
                    # Use the image path provided by Unstructured (images are saved automatically, e.g., in the figures folder)
                    image_path = getattr(element.metadata, "image_path", None)
                    
                    if image_path:
                        if os.path.isdir(image_path):
                            images = [f for f in os.listdir(image_path) if f.endswith(".jpg")]
                            if images:
                                image_file = os.path.join(image_path, images[0])
                            else:
                                print(f"⚠️ No JPG images found in directory for {element.category}")
                                continue
                            
                        elif os.path.isfile(image_path):
                            image_file = image_path
                        else:
                            print(f"⚠️ Invalid image path for {element.category} on {page_key}")
                            continue
                        if os.path.exists(image_file):
                            try:
                                with open(image_file, "rb") as f:
                                    regenerated_b64 = base64.b64encode(f.read()).decode("utf-8")
                                self.image_metadata.setdefault(page_key, []).append({
                                    "path": image_file,
                                    "base64": regenerated_b64
                                })
                            except Exception as e:
                                print(f"⚠️ Failed to process image for {page_key}: {e}")
                        else:
                            print(f"⚠️ No valid image_path found for {element.category} on {page_key}")
                            
            # print for confirmation                
            print(f"✅ Extracted content from: {pdf_file}")

    def chunk_and_embed(self):
        text_elements = [el for el in self.elements if el.category in ["NarrativeText", "Title", "Text", "List", "Table"]]
        if not text_elements:
            print("⚠️ No text elements to chunk!")
            return
        chunks = chunk_elements(text_elements, max_characters=2000, new_after_n_chars=2000, overlap=200)
        chunk_texts = [c.text for c in chunks if c.text]
        if not chunk_texts:
            print("⚠️ No text chunks produced!")
            return
        embeddings = embed_documents(chunk_texts)
        coll = self.chroma_client.get_or_create_collection(name="l3.2_docs")
        ids, metadata_list = [], []
        for i, chunk in enumerate(chunks):
            orig = getattr(chunk.metadata, "orig_elements", [])
            meta = {"orig_elements": str([el.metadata.__dict__ for el in orig])}
            doc_id = f"chunk:{i}"
            ids.append(doc_id)
            metadata_list.append(meta)
        coll.add(ids=ids, documents=chunk_texts, metadatas=metadata_list, embeddings=embeddings)
        print(f"📊 Indexed {len(ids)} doc chunks into Chroma.")

    def process_pdfs(self):
        self.extract_content()
        self.chunk_and_embed()
        print("✅ Finished unstructured ETL!")
        
import json

def extract_final_json(text: str) -> str:
    try:
        json_output = json.loads(text)
        return json_output.get("final_answer", "").strip()
    except Exception:
        return text.strip()

class ImageMetadataExtractor:
    """
    Stores image metadata in a separate Chroma collection.
    """
    def __init__(self, db_path="chroma_db"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.image_meta_collection = self.chroma_client.get_or_create_collection(name="image_meta_docs")
        self.arch_components_llm = Ollama(model="llama3.2-vision", temperature=0.0)
        self.compliance_info_llm = Ollama(model="llama3.2-vision", temperature=0.0)
        self.dimension_labels_llm = Ollama(model="llama3.2-vision", temperature=0.0)
        self.accessibility_llm = Ollama(model="llama3.2-vision", temperature=0.0)
        self.table_llm = Ollama(model="llama3.2-vision", temperature=0.0)

    def embed_text(self, text: str) -> list:
        emb = get_text_embedding_for_query(text)
        return emb.tolist() if isinstance(emb, torch.Tensor) else emb

    def extract_arch_components(self, base64_image: str) -> str:
        prompt = (
            """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert architectural component identifier for NZ Building Code images.
            Identify all visible components. if there are no visible components, say "None".

            Report only what is visible; do not infer.
            <|end_of_text_id|>
            """
        )
        response = self.arch_components_llm.generate(
            prompts=[prompt],
            images=[base64_image]
        )
        full_text = response.generations[0][0].text
        return extract_final_json(full_text)

    def extract_dimension_labels(self, base64_image: str) -> str:
        prompt = (
            """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a dimension labels extractor for NZ building code diagrams.
            List all numeric and textual dimension labels visible in the image and explain what each represents.
            If there are no visible dimension labels, say "None".
            Report only what is shown; do not guess.
            If none are present, say "None".
            <|end_of_text_id|>
            """
        )
        response = self.dimension_labels_llm.generate(
            prompts=[prompt],
            images=[base64_image]
        )
        full_text = response.generations[0][0].text
        return extract_final_json(full_text)


    def extract_compliance_info(self, base64_image: str) -> str:
        # Agent 3: Extract compliance-related details.
        prompt = (
            """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert in NZ Building Code compliance.
            Extract all visible compliance details such as minimum or maximum measurements and constraints.
            If there are no visible compliance details, say "None".

            Use only NZ Building Code standards and report only what is visible; do not guess.
            <|end_of_text_id|>
            """
        )
        response = self.compliance_info_llm.generate(
            prompts=[prompt],
            images=[base64_image]
        )
        full_text = response.generations[0][0].text
        return extract_final_json(full_text)

    def extract_accessibility_features(self, base64_image: str) -> str:
        # Agent 4: Identify accessibility and safety features.
        prompt = (
            """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an accessibility and safety feature identifier for NZ architectural diagrams.
            Identify all visible accessibility and safety elements.
            If there are no visible accessibility and safety elements, say "None".

            Report only what is visible.
            <|end_of_text_id|>
            """
        )
        response = self.accessibility_llm.generate(
            prompts=[prompt],
            images=[base64_image]
        )
        full_text = response.generations[0][0].text
        return extract_final_json(full_text)


    def extract_table_data(self, base64_image: str) -> str:
        # Agent 7: Extract any tabular data from the image.
        prompt = (
            """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a table data extractor.
            Extract any tabular data visible in the image in a structured format.
            If there are no visible table data, say "None"
            Provide only the table data without extra commentary.
            <|end_of_text_id|>
            """
        )
        response = self.table_llm.generate(
            prompts=[prompt],
            images=[base64_image]
        )
        full_text = response.generations[0][0].text
        return extract_final_json(full_text)

    def store_image_description(self, doc_id: str, meta: dict):
        text_desc = f"Image at {meta.get('image_path', 'unknown location')}"
        emb = self.embed_text(text_desc)
        base64_image = meta.get("base64", "")
        if base64_image:
            arch_components = self.extract_arch_components(base64_image)
            dimension_labels = self.extract_dimension_labels(base64_image)
            compliance_info = self.extract_compliance_info(base64_image)
            accessibility_features = self.extract_accessibility_features(base64_image)
            table_data = self.extract_table_data(base64_image)
            final_summary = (
                "## Architectural Components\n" + arch_components + "\n\n" +
                "## Dimension Labels\n" + dimension_labels + "\n\n" +
                "## Compliance Information\n" + compliance_info + "\n\n" +
                "## Accessibility & Safety Features\n" + accessibility_features + "\n\n" +
                "## Table Data\n" + table_data + "\n"
            )
            meta["image_descriptions"] = final_summary
        else:
            meta["image_descriptions"] = "No image description extracted"

        self.image_meta_collection.upsert(
            ids=[doc_id],
            embeddings=[emb],
            documents=[text_desc],
            metadatas=[meta]
        )
        print(f"Stored metadata for image: {doc_id}")

    def run_extraction_for_all_pages(self, rag):
        existing = self.image_meta_collection.get()
        existing_ids = set(existing["ids"])
        for page_key, image_list in rag.image_metadata.items():
            for idx, item in enumerate(image_list):
                doc_id = f"{page_key}_{idx}"
                if doc_id in existing_ids:
                    continue
                meta = {
                    "page_key": page_key,
                    "image_path": item["path"],
                    "base64": item.get("base64", "")
                }
                self.store_image_description(doc_id, meta)

# -------------------------------------------------
# 4. Retrieval Functions
# -------------------------------------------------
def retrieve_text_docs(query_text: str, top_k: int = 1):
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    coll = chroma_client.get_or_create_collection(name="l3.2_docs")
    query_emb = get_text_embedding_for_query(query_text).tolist()
    results = coll.query(query_embeddings=[query_emb], n_results=top_k)
    doc_texts = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    out_docs = []
    for dtext, meta, dist in zip(doc_texts, metas, distances):
        new_meta = {**meta, "distance": dist}
        out_docs.append(Document(page_content=dtext, metadata=new_meta))
    if out_docs:
        combined_text = "\n\n".join(doc.page_content for doc in out_docs)
        return combined_text
    else:
        return "No relevant documents found."

def retrieve_image_meta(query_text: str, top_k: int = 1):
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    coll = chroma_client.get_or_create_collection(name="image_meta_docs")
    query_emb = get_text_embedding_for_query(query_text).tolist()
    results = coll.query(query_embeddings=[query_emb], n_results=top_k)
    doc_texts = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    out_docs = []
    for dtext, meta, dist in zip(doc_texts, metas, distances):
        new_meta = {**meta, "distance": dist}
        # Use the stored image description if available.
        out_docs.append(Document(page_content=meta.get("image_descriptions", "No description"), metadata=new_meta))
    if out_docs:
        combined_text = "\n\n".join(doc.page_content for doc in out_docs)
        return combined_text
    else:
        return "No relevant image metadata found."

# -------------------------------------------------
# 5. Prompt Templates
# -------------------------------------------------
text_retrieval_prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""\
You are an expert assistant specialized in New Zealand building codes. Your task is to retrieve relevant text from building code documents.
Once you have gathered the info, output exactly one JSON object:
{
  "final_text": "...summarized text..."
}
NO extra text outside the JSON.
Question: {input}
{agent_scratchpad}
"""
)

image_retrieval_prompt_template_v2 = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""\
You are an expert assistant specialized in New Zealand building codes. Your task is to retrieve relevant image metadata from building code documents based on a text query.
Once you have gathered the info, output exactly one JSON object:
{
  "final_image_data": "<summary of image metadata>"
}
NO extra text outside the JSON.
Question: {input}
{agent_scratchpad}
"""
)

# New prompt for the final answer agent.
final_answer_prompt_template = PromptTemplate(
    input_variables=["retrieval", "question"],
    template="""\
You are an expert building code assistant for New Zealand. You have been provided with the following combined retrieved information from building code documents:

{retrieval}

Based on this, answer the following question as concisely as possible:
{question}

Output exactly one JSON object with:
  "final_answer": <the final answer>
No extra commentary.
"""
)

# -------------------------------------------------
# 6. Tools & Agents
# -------------------------------------------------
from langchain.agents import Tool

text_tool = Tool(
    name="retrieve_text_docs",
    func=retrieve_text_docs,
    description="Retrieve building code text from 'l3.2_docs'."
)

image_meta_tool = Tool(
    name="retrieve_image_metadata",
    func=retrieve_image_meta,
    description="Retrieve building code image metadata from 'image_meta_docs'."
)

llm_for_text_agent = Ollama(model="llama3.1", temperature=0.0)
text_retrieval_agent = initialize_agent(
    tools=[text_tool],
    llm=llm_for_text_agent,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True,
    prompt_template=text_retrieval_prompt_template
)

llm_for_image_retrieval_agent = Ollama(model="llama3.1", temperature=0.0)
image_retrieval_agent = initialize_agent(
    tools=[image_meta_tool],
    llm=llm_for_image_retrieval_agent,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True,
    prompt_template=image_retrieval_prompt_template_v2
)

# New agent solely for generating the final answer.
llm_for_answer_agent = Ollama(model="llama3.1", temperature=0.0)
final_answer_agent = initialize_agent(
    tools=[],
    llm=llm_for_answer_agent,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True,
    prompt_template=final_answer_prompt_template
)

# -------------------------------------------------
# 7. Graph Nodes
# -------------------------------------------------
class GraphState(TypedDict):
    chat_history: List[Dict[str, str]]
    memory: List[Dict[str, str]]
    question: str
    text_retrieval: str
    image_retrieval: str
    combined_retrieval: str
    final_answer: str
    image_path: Optional[str]
    input: str

def node_init_memory(state: GraphState) -> GraphState:
    if not state.get("memory"):
        state["memory"] = []
    if not state.get("chat_history"):
        state["chat_history"] = []
    state["memory"].append({"role": "user", "content": state["question"]})
    return state

def node_text_retrieval(state: GraphState) -> GraphState:
    conversation_text = "\n".join(
        [f"{msg['role'].upper()}: {msg['content']}" for msg in state["memory"]]
    )
    response = text_retrieval_agent.run({
        "input": conversation_text,
        "agent_scratchpad": "",
        "chat_history": state["chat_history"]
    })
    try:
        parsed = json.loads(response)
    except Exception:
        parsed = {}
    
    if "final_text" in parsed:
        state["text_retrieval"] = parsed["final_text"]
    else:
        state["text_retrieval"] = response

    state["memory"].append({"role": "assistant", "content": response})
    return state

def node_image_retrieval(state: GraphState) -> GraphState:
    query_for_images = state["question"]
    response_data = image_retrieval_agent.run({
        "input": query_for_images,
        "agent_scratchpad": "",
        "chat_history": state["chat_history"]
    })
    
    try:
        parsed = json.loads(response_data)
    except Exception:
        parsed = {}
    
    final_image_data = parsed.get("final_image_data", "")
    if isinstance(final_image_data, list):
        final_image_data = "\n".join(
            [item.get("image_descriptions", "") if isinstance(item, dict) else str(item) for item in final_image_data]
        )
    elif not isinstance(final_image_data, str):
        final_image_data = str(final_image_data)
    
    state["image_retrieval"] = final_image_data
    return state

# New node: concatenate the text and image retrieval outputs.
def node_concatenate_retrieval(state: GraphState) -> GraphState:
    text_data = state.get("text_retrieval", "")
    image_data = state.get("image_retrieval", "")
    combined = text_data + "\n\n" + image_data
    state["combined_retrieval"] = combined
    state["memory"].append({"role": "assistant", "content": f"Combined retrieval:\n{combined}"})
    return state

# New node: final answer generation using the combined retrieval text.
def node_final_answer(state: GraphState) -> GraphState:
    input_str = final_answer_prompt_template.format(
        retrieval=state.get("combined_retrieval", ""),
        question=state["question"]
    )
    final_response = final_answer_agent.run({
        "input": input_str,
        "agent_scratchpad": "",
        "chat_history": state["chat_history"]
    })
    
    # Ensure we extract the final answer as a string.
    if isinstance(final_response, dict):
        final_answer_str = final_response.get("final_answer", json.dumps(final_response))
    elif isinstance(final_response, str):
        try:
            parsed_final = json.loads(final_response)
            final_answer_str = parsed_final.get("final_answer", final_response)
        except Exception:
            final_answer_str = final_response
    else:
        final_answer_str = str(final_response)
    
    state["final_answer"] = final_answer_str
    state["memory"].append({"role": "assistant", "content": final_answer_str})
    return state

# -------------------------------------------------
# 8. Workflow Compilation
# -------------------------------------------------
workflow = StateGraph(state_schema=GraphState)
workflow.add_node("init_memory", node_init_memory)
workflow.add_node("text_retrieval_node", node_text_retrieval)
workflow.add_node("image_retrieval_node", node_image_retrieval)
workflow.add_node("concatenate_node", node_concatenate_retrieval)
workflow.add_node("final_answer_node", node_final_answer)

workflow.add_edge(START, "init_memory")
workflow.add_edge("init_memory", "text_retrieval_node")
workflow.add_edge("text_retrieval_node", "image_retrieval_node")
workflow.add_edge("image_retrieval_node", "concatenate_node")
workflow.add_edge("concatenate_node", "final_answer_node")
workflow.add_edge("final_answer_node", END)
app = workflow.compile()


# -------------------------------------------------
# 8. Streamlit UI
# -------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "image_meta_extracted" not in st.session_state:
    st.session_state.image_meta_extracted = False

st.set_page_config(page_title="NZ Building Code RAG", layout="centered")
custom_css = """
<style>
.chat-bubble {
  border-radius: 8px;
  padding: 12px;
  margin: 8px 0;
  max-width: 80%;
  line-height: 1.4;
  color: #000;
  overflow-wrap: break-word;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.user-bubble {
  background-color: #CCE6FF;
  align-self: flex-start;
}
.assistant-bubble {
  background-color: #E9E9E9;
  align-self: flex-end;
}
.chat-container {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-top: 1rem;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.sidebar.header("Conversations")
if st.sidebar.button("New Chat"):
    st.session_state.chat_history = []
    st.rerun()

st.title("CodeVision NZ: Intelligent Building Code Retrieval Assistant")
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"""<div class="chat-bubble user-bubble"><strong>User:</strong> {msg['content']}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="chat-bubble assistant-bubble"><strong>Assistant:</strong> {msg['content']}</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    question = st.text_input("Enter your question", key="question_input")
    uploaded_image = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"], key="image_upload")
    submitted = st.form_submit_button("Send")
    if submitted and question:
        image_path = None
        if uploaded_image is not None:
            temp_dir = "temp_images"
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, uploaded_image.name)
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            image_path = temp_image_path

        # Update the initial state to include the new key "combined_retrieval"
        init_state: GraphState = {
            "chat_history": st.session_state.chat_history.copy(),
            "memory": st.session_state.chat_history.copy(),
            "question": question,
            "final_answer": "",
            "image_path": image_path,
            "input": question,
            "text_retrieval": "",
            "image_retrieval": "",
            "combined_retrieval": ""
        }
        with st.spinner("Generating answer..."):
            result_state = app.invoke(init_state)
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": result_state["final_answer"]})
        st.rerun()

# -------------------------------------------------
# 9. PDF / Image Preprocessing
# -------------------------------------------------
if not st.session_state.pdf_processed:
    with st.spinner("Processing PDFs for text-based RAG..."):
        rag = RAG_embedding(
            folder_path=r"C:\Users\GGPC\Desktop\Work\Personal Projects\AI_RAG_NZ_BuildingCOde\DATA",
            prefix="NZ_Building_Code_",
            suffix=".pdf",
            db_path="chroma_db",
            use_unstructured=True
        )
        rag.process_pdfs()
        st.session_state.rag = rag
    st.session_state.pdf_processed = True

if not st.session_state.image_meta_extracted:
    if not st.session_state.rag:
        st.warning("No RAG object found. Please re-run.")
    else:
        with st.spinner("Storing image metadata..."):
            extractor = ImageMetadataExtractor(db_path="chroma_db")
            extractor.run_extraction_for_all_pages(st.session_state.rag)
        st.session_state.image_meta_extracted = True

st.write("All PDFs processed. You can now query the system!")
