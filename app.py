import streamlit as st
import asyncio
import chromadb

from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    ReActAgent,
    ToolCallResult,
    AgentStream,
)

# Initialize LLM and embedding model
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
embed_model = HuggingFaceInferenceAPIEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Set up vector store
db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Set up query engine
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=embed_model
)
query_engine = index.as_query_engine(llm=llm)
query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="personas",
    description="descriptions for various types of personas",
    return_direct=False,
)

# Define arithmetic tools
def add(a: int, b: int) -> int:
    "add two numbers"
    return a + b

def subtract(a: int, b: int) -> int:
    "subtract two numbers"
    return a - b

def multiply(a: int, b: int) -> int:
    "multiply two numbers"
    return a * b

def divide(a: int, b: int) -> int:
    "divide two numbers"
    return a / b

# Define agents
calculator_agent = ReActAgent(
    name="calculator",
    description="performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. Use your tools for any math operations.",
    tools=[add, subtract, multiply, divide],
    llm=llm,
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about math problems",
    system_prompt="Use your tool to query a RAG system to answer information about math problems.",
    tools=[query_engine_tool],
    llm=llm,
)

agent = AgentWorkflow(agents=[calculator_agent, query_agent], root_agent="calculator")

# Streamlit UI
st.title("🧠 Multi-Agent RAG + Calculator")
user_msg = st.text_input("Ask a question (e.g., 'What is 8 * 7?' or 'Tell me about algebra'):")

if user_msg:
    with st.spinner("Thinking..."):
        async def run_agent():
            handler = agent.run(user_msg)
            async for ev in handler.stream_events():
                if isinstance(ev, ToolCallResult):
                    st.info(f"📦 Tool called: `{ev.tool_name}` with `{ev.tool_kwargs}` → **{ev.tool_output}**")
                # Skipping AgentStream (no live thinking output)
            final = await handler
            return final

        result = asyncio.run(run_agent())
        st.success(f"✅ Final Answer: {result}")
