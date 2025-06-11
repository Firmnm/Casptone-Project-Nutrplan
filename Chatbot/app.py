# app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Import fungsi dan objek dari nutrisi_model.py
from agent import answer_query, llm, retrievers

app = FastAPI(title="Nutrisi Assistant API")

# Aktifkan CORS untuk akses frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ubah sesuai kebutuhan keamanan
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Skema data untuk request dan response
class QuestionRequest(BaseModel):
    question: str

class DocumentResponse(BaseModel):
    source: str
    content: str

class AnswerResponse(BaseModel):
    answer: str
    documents: List[DocumentResponse]

# Endpoint utama
@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    try:
        answer, docs = answer_query(req.question, llm, retrievers)
        doc_responses = [
            DocumentResponse(source=doc.metadata.get("source", "unknown"), content=doc.page_content)
            for doc in docs
        ]
        return AnswerResponse(answer=answer, documents=doc_responses)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
