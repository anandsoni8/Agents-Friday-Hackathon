import os
from typing import List, Dict, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    judge_scores: Dict[str, float]
    is_relevant: bool

# --- MODEL INITIALIZATION ---
def get_llm():
    # You can change the model to "deepseek-v3" if your hackathon provides a custom endpoint
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-ada-002")

# --- PDF PROCESSING ---
def process_pdf_to_chroma(file_path: str, persist_directory="./chroma_index"):
    """Loads a PDF, chunks it, and saves it to ChromaDB."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=get_embeddings(),
        persist_directory=persist_directory
    )
    return vectorstore

# --- AGENT 1: RETRIEVER ---
def retrieve_agent(state: AgentState):
    question = state["question"]
    db = Chroma(persist_directory="./chroma_index", embedding_function=get_embeddings())
    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    return {"documents": [doc.page_content for doc in docs]}

# --- AGENT 2: CONTEXT GRADER ---
def context_grader_agent(state: AgentState):
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()
    
    prompt = f"""You are a strict relevance grader. Does the following context contain information to answer the question?
    Question: {question}
    Context: {documents}
    Answer purely with 'yes' or 'no'."""
    
    response = llm.invoke(prompt).content.strip().lower()
    is_relevant = "yes" in response
    return {"is_relevant": is_relevant}

# --- ROUTING LOGIC ---
def decide_to_generate(state: AgentState):
    if state["is_relevant"]:
        return "synthesizer"
    else:
        return "reject_out_of_context"

def reject_out_of_context(state: AgentState):
    return {"generation": "I cannot answer this. The provided PDF does not contain context to answer this query."}

# --- AGENT 3: SYNTHESIZER ---
def synthesizer_agent(state: AgentState):
    llm = get_llm()
    prompt = f"""Use the following context to answer the question.
    Context: {state['documents']}
    Question: {state['question']}
    Answer strictly based on the context."""
    
    response = llm.invoke(prompt)
    return {"generation": response.content}

# --- AGENT 4: EVALUATOR ---
def evaluator_agent(state: AgentState):
    if not state.get("is_relevant", False):
        return {"judge_scores": {"Faithfulness": 0.0, "Relevance": 0.0, "Accuracy": 0.0, "Safety": 1.0}}
    
    # Mocking evaluator scores for dashboard visualization
    scores = {"Faithfulness": 0.95, "Relevance": 0.98, "Accuracy": 0.92, "Safety": 1.0}
    return {"judge_scores": scores}

# --- BUILD GRAPH ---
workflow = StateGraph(AgentState)

workflow.add_node("retriever", retrieve_agent)
workflow.add_node("context_grader", context_grader_agent)
workflow.add_node("reject_out_of_context", reject_out_of_context)
workflow.add_node("synthesizer", synthesizer_agent)
workflow.add_node("evaluator", evaluator_agent)

workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "context_grader")
workflow.add_conditional_edges(
    "context_grader", 
    decide_to_generate,
    {"synthesizer": "synthesizer", "reject_out_of_context": "reject_out_of_context"}
)
workflow.add_edge("synthesizer", "evaluator")
workflow.add_edge("reject_out_of_context", "evaluator")
workflow.add_edge("evaluator", END)

agent_pipeline = workflow.compile()

def process_query_with_agents(query: str):
    final_state = agent_pipeline.invoke({"question": query})
    return final_state["generation"], final_state["judge_scores"]