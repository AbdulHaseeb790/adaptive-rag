from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END,START
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_core.documents import Document
import os 
load_dotenv()
llm=ChatGroq(model='openai/gpt-oss-120b')
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
loader = PyPDFLoader(r'C:\Users\SOHAIL\adaptive_rag\attention.pdf')
docs=loader.load()
text=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
chunks=text.split_documents(docs)
vector_store=FAISS.from_documents(chunks,embedding)
retriver=vector_store.as_retriever()
class state(TypedDict):
    question:str
    documents:list[str]
    generation:str
    grounded: str 
    useful: str
def retrieve(state):
    question=state['question']
    documents=retriver.invoke(question)
    return {'documents': documents, 'question': question}
class GradeDocuments(BaseModel):
    score: str = Field(description="Documents is relevant to question? Answer 'yes' or 'no' in JSON format with key 'score'")

class GradeGrounded(BaseModel):
    score: str = Field(description="Is the answer grounded in the documents? Answer 'yes' or 'no' in JSON format with key 'score'")

class GradeUseful(BaseModel):
    score: str = Field(description="Does the answer address the question? Answer 'yes' or 'no' in JSON format with key 'score'")
class RouteQuery(BaseModel):
    datasource: str = Field(description="Choose one: 'vectorstore', 'web_search', or 'llm_direct'")
def question_router(state: dict) -> str:
    question = state["question"]
    router = llm.with_structured_output(RouteQuery, method="json_mode")
    result = router.invoke(f"""You are a router. Based on the question, choose where to route it.
- 'llm_direct' → simple general knowledge question
- 'vectorstore' → question about the document
- 'web_search' → needs recent internet data

Question: {question}
Respond in JSON with exactly this format: {{"datasource": "vectorstore"}}""")
    return result.datasource
def llm_direct(state: dict) -> dict:
    question = state["question"]
    generation = llm.invoke(question)
    return {"generation": generation.content, "question": question, "documents": []}
def grade_node(state):
    question=state['question']
    documents=state['documents']
    grader = llm.with_structured_output(GradeDocuments, method="json_mode")
    relevant_docs = []  
    for doc in documents:
        result = grader.invoke(f"Question: {question}\nDocument: {doc.page_content}\nRespond in JSON with exactly this format: {{\"score\": \"yes\"}} or {{\"score\": \"no\"}}")
        if result.score == "yes":  # if LLM says relevant
            relevant_docs.append(doc)  # keep this doc
    return {"documents": relevant_docs, "question": question}

web_search_tool=TavilySearchResults(k=3)
def web_search(state:dict)->dict:
    question=state['question']
    docs=state.get('documents',[])  
    results=web_search_tool.invoke({'query':question})
    web_docs = [Document(page_content=r["content"]) for r in results]
    return {'documents':docs+web_docs,'question':question}
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
rag_prompt=ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you don't know the answer, say you don't know.
Context: {context}
Question:{question}
Answer.

""")
def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)
rag_chain = rag_prompt | llm | StrOutputParser()
def generate(state: dict) -> dict:
    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke({
        "context": format_docs(documents),
        "question": question
    })
    return {"generation": generation, "documents": documents, "question": question}
def grade_generation(state:dict)->dict:
    question=state['question']
    documents=state['documents']
    generation=state['generation']
    grader = llm.with_structured_output(GradeGrounded, method="json_mode")
    result = grader.invoke(f"Documents: {format_docs(documents)}\nAnswer: {generation}\nRespond in JSON with exactly this format: {{\"score\": \"yes\"}} or {{\"score\": \"no\"}}")

    return {"question": question, "documents": documents, "generation": generation, "grounded": result.score}
def grade_usefulness(state: dict) -> dict:
    question = state["question"]
    generation = state["generation"]
    grader = llm.with_structured_output(GradeUseful, method="json_mode")
    result = grader.invoke(f"Question: {question}\nAnswer: {generation}\nRespond in JSON with exactly this format: {{\"score\": \"yes\"}} or {{\"score\": \"no\"}}")
    return {"question": question, "documents": state["documents"], "generation": generation, "useful": result.score}
def route_after_generation(state: dict) -> str:
    grounded = state["grounded"]
    useful = state["useful"]
    if grounded == "yes" and useful == "yes":
        return "END"
    elif grounded == "no":
        return "generate"
    else:
        return "retrieve"
def route_after_grade(state: dict) -> str:
    documents = state["documents"]
    if not documents:
        print("→ going to WEB SEARCH")
        return "web_search"
    print("→ going to GENERATE from PDF docs")
    return "generate"
workflow = StateGraph(state)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_node)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)
workflow.add_node("grade_generation", grade_generation)
workflow.add_node("grade_usefulness", grade_usefulness)
workflow.add_node("llm_direct", llm_direct)


workflow.add_conditional_edges(
    START,
    question_router,
    {
        "vectorstore": "retrieve",
        "web_search": "web_search",
        "llm_direct": "llm_direct"
    }
)
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    route_after_grade,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("llm_direct", END)
workflow.add_edge("generate", "grade_generation")
workflow.add_edge("grade_generation", "grade_usefulness")
workflow.add_conditional_edges(
    "grade_usefulness",
    route_after_generation,
    {
        "END": END,
        "generate": "generate",
        "retrieve": "retrieve"
    }
)

app = workflow.compile()
result = app.invoke({"question": "What did OpenAI release this week??"})
print(result["generation"])