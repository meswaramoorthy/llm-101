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
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

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

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    # if "vector_store" in st.session_state:
    #     vector_store = st.session_state["vector_store"]
    vector_store = load_docs_and_create_store()
    retrieved_docs = vector_store.similarity_search(query, k=5, fetch_k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    if "llm" in st.session_state:
        llm = st.session_state["llm"]
    else:
         raise Exception("llm not found")
    
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

# Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If the information isn’t available in the context to formulate an answer, simply reply that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    if "llm" in st.session_state:
        llm = st.session_state["llm"]
    else:
         raise Exception("llm not found")

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

def getGraph():
    if "graph" not in st.session_state:
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node(query_or_respond)
        graph_builder.add_node(ToolNode([retrieve]))
        # graph_builder.remove_node(generate)
        graph_builder.add_node(generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        memory = MemorySaver()

        graph = graph_builder.compile(checkpointer=memory)
        st.session_state["graph"] = graph

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
     
     # Specify an ID for the thread
    config = {"configurable": {"thread_id": "abc123"}}
    final_result =[]
    if submitted:
        for step in graph.stream(
            {"messages": [{"role": "user", "content": text}]},
            stream_mode="values",
            config=config,
        ):
            final_result = step
        
        st.info(final_result["messages"][-1])

if __name__ == "__main__":
    main()