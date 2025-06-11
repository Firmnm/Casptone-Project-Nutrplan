from fastapi import FastAPI
from pydantic import BaseModel
from model import load_model
from generator2 import generate_diet_program
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti "" dengan asal frontend jika ingin lebih aman
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
tokenizer, model = load_model()

class UserInfo(BaseModel):
    goal: str
    duration: str
    age: int
    weight: int
    height: int
    eatingPattern: str
    allergies: str
    dislikes: str
    exerciseFrequency: str
    sleepQuality: str
    

@app.post("/generate-diet")
def generate_diet(user_info: UserInfo):
    result = generate_diet_program(user_info.dict(), tokenizer, model)
    return {"result": result}
