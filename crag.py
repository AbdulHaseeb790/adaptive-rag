from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
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
loader = PyPDFLoader(r'C:\Users\SOHAIL\adaptive_rag\Google AI Essentials Specialization Solution.pdf')
docs=loader.load()
text=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
chunks=text.split_documents(docs)
vector_store=FAISS.from_documents(chunks,embedding)
retriver=vector_store.as_retriever()
class state(TypedDict):
    question:str
    documents:list[str]
    generation:str
def retrieve(state):
    question=state['question']
    documents=retriver.invoke(question)
    return {'documents': documents, 'question': question}
class GradeDocuments(BaseModel):
    score:str=Field(description="documents is revelant to question? 'yes'or 'no'")
def grade_node(state):
    question=state['question']
    documents=state['documents']
    grader=llm.with_structured_output(GradeDocuments)
    relevant_docs = []  
    for doc in documents:
        result = grader.invoke(f"Question: {question}\nDocument: {doc.page_content}")  # ask LLM to grade
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
def route_after_grade(state: dict) -> str:
    documents = state["documents"]
    if not documents:
        return "web_search"
    return "generate"

workflow = StateGraph(state)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_node)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
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
workflow.add_edge("generate", END)

app = workflow.compile()
result = app.invoke({"question": "What is machine learning?"})
print(result["generation"])
class GradeGrounded(BaseModel):
    score:str=Field(description="is the answer grounded in document 'yes'or'no'")
class GradeUseful(BaseModel):
    score:str=Field(description="does the answer actually address the question"'yes'or'no')
    
          


