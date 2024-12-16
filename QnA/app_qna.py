import streamlit as st
import getpass
import os
import bs4
import json
from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Initialize LLM
@st.cache_resource
def initialize_llm():
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment="gpt-4o-mini",
        openai_api_version="2024-02-15-preview",
    )
    st.session_state["llm"] = llm
    return llm

@st.cache_resource
def load_docs_and_create_store():
    with open('article_links.json', 'r') as f:
        article_links = json.load(f)
    # Load and chunk contents of the blog
    loader = WebBaseLoader(
        web_paths=(list(article_links.values())),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    print(f"Split blog post into {len(all_splits)} sub-documents.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore(embeddings)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)
    st.session_state["vector_store"] = vector_store
    return vector_store

# Define application steps
def retrieve(state: State):
    if "vector_store" in st.session_state:
        vector_store = st.session_state["vector_store"]
    else:
         raise Exception("Vector Store not found")
    
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Define prompt for question-answering
    prompt = hub.pull("rlm/rag-prompt")
    if "llm" in st.session_state:
        llm = st.session_state["llm"]
    else:
         raise Exception("llm not found")

    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def getGraph():
    if "graph" not in st.session_state:
        graph_builder = StateGraph(State)
        # Compile application and test
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        st.session_state["graph"] = graph

        response = graph.invoke({"question": "What is Task Decomposition?"})
        print(response["answer"])
    return st.session_state["graph"]

def main():
    st.title("Q&A Bot Powered by LLM")
    
    initialize_llm()
    load_docs_and_create_store()
    graph = getGraph()

    with st.form("my_form"):
        text = st.text_area(
            "Enter text:",
        )
        submitted = st.form_submit_button("Submit")
     
    if submitted:
        response = graph.invoke({"question": text})
        st.info(response["answer"])

if __name__ == "__main__":
    main()