import streamlit as st
import os
import tempfile
from rag_core import process_pdf_to_chroma, process_query_with_agents

st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")

st.title("🤖 Multi-Agent AI Assistant")

# Sidebar Configuration
with st.sidebar:
    st.header("1. API Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
    st.header("2. Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")
    
    if uploaded_file and st.button("Process Document"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API Key first.")
        else:
            with st.spinner("Agent generating Vector Database..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                process_pdf_to_chroma(tmp_path)
                st.session_state["doc_processed"] = True
                st.success("Database ready! You can now chat.")

    st.header("3. LLM Observability")
    st.caption("Agent-Evaluated Metrics")
    if "scores" in st.session_state:
        scores = st.session_state["scores"]
        st.metric("Faithfulness", f"{scores.get('Faithfulness', 0) * 100}%")
        st.metric("Relevance", f"{scores.get('Relevance', 0) * 100}%")
        st.metric("Accuracy", f"{scores.get('Accuracy', 0) * 100}%")
        st.metric("Safety", f"{scores.get('Safety', 0) * 100}%")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Please upload a PDF in the sidebar to get started."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question about the document..."):
    if not st.session_state.get("doc_processed"):
        st.warning("Please upload and process a PDF first!")
        st.stop()
        
    # Show user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Process query with Agents
    with st.spinner("Multi-Agent Team is working on your answer..."):
        answer, scores = process_query_with_agents(prompt)
        st.session_state["scores"] = scores
        
    # Show assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
    st.rerun() # Refresh the page to update the dashboard metrics