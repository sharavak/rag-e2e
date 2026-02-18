from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily import TavilyClient

load_dotenv()

# ── Init ──────────────────────────────────────────────────────
tavily = TavilyClient(api_key=os.getenv("Tavily_key"))

def build_faiss_vectorstore():
    loader = TextLoader("upload_notes.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)

print("Building vector store...")
vector_store = build_faiss_vectorstore()
print("Vector store ready!")

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ── Tools ─────────────────────────────────────────────────────
@tool
def retrieve_from_document(query: str) -> str:
    """Strictly retrieve information from the uploaded notes only."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    if not retrieved_docs:
        return "NO_DOCUMENT_MATCH"
    return "\n\n".join([doc.page_content for doc in retrieved_docs])

@tool
def web_search(query: str) -> str:
    """Use Tavily only when document has no relevant information."""
    response = tavily.search(query=query, max_results=3)
    if not response or not response.get("results"):
        return "NO_WEB_RESULT"
    return "\n\n".join(
        f"- {item['title']}: {item['content']}" for item in response["results"]
    )

tools = [retrieve_from_document, web_search]

# ── Prompt ────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an Agentic RAG assistant.
STRICT RULES:
1. ALWAYS check the document first using retrieve_from_document tool.
2. If document returns "NO_DOCUMENT_MATCH", then only use web_search.
3. If both fail, respond: "The information is not available."
4. Do NOT hallucinate.
5. Prefer document answers over web answers.
"""
    ),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# ── Agent ─────────────────────────────────────────────────────
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=8,
    handle_parsing_errors=True,
)

# ── FastAPI ───────────────────────────────────────────────────
app = FastAPI(title="Agentic RAG API")
chat_histories = {}

class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"

class AnswerResponse(BaseModel):
    question: str
    answer: str
    session_id: str

@app.get("/")
def root():
    return {"message": "Agentic RAG API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if request.session_id not in chat_histories:
        chat_histories[request.session_id] = []

    history = chat_histories[request.session_id]

    result = agent_executor.invoke({
        "input": request.question,
        "chat_history": history,
    })

    ai_text = result["output"]

    history.append(HumanMessage(content=request.question))
    history.append(AIMessage(content=ai_text))

    return AnswerResponse(
        question=request.question,
        answer=ai_text,
        session_id=request.session_id
    )

@app.delete("/clear/{session_id}")
def clear_history(session_id: str):
    if session_id in chat_histories:
        del chat_histories[session_id]
    return {"message": f"History cleared for session: {session_id}"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8010, reload=False)
