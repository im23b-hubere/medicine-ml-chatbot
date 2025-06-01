import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Chatverlauf im Speicher (nur für Demo, nicht persistent)
chat_history = []
last_confidence = None
last_matched_question = None

# Embedding-Modell und Daten laden
with open(os.path.join(BASE_DIR, 'medquad_full.json'), 'r', encoding='utf-8') as f:
    faq_data = json.load(f)
questions = [item['question'] for item in faq_data]
answers = [item['answer'] for item in faq_data]

model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(questions, show_progress_bar=True, convert_to_numpy=True)

def retrieve_answer(user_question):
    user_emb = model.encode([user_question], convert_to_numpy=True)
    sims = cosine_similarity(user_emb, question_embeddings)[0]
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_question = questions[best_idx]
    answer = answers[best_idx]
    return answer, best_score, best_question

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("terminal_chat.html", {
        "request": request,
        "chat_history": chat_history,
        "accuracy": None,  # Nicht mehr relevant
        "last_confidence": last_confidence,
        "last_matched_question": last_matched_question,
    })

@app.post("/chat", response_class=HTMLResponse)
def chat(request: Request, user_input: str = Form(...)):
    global last_confidence, last_matched_question
    answer, conf, matched_q = retrieve_answer(user_input)
    chat_history.append((user_input, answer, conf, matched_q))
    last_confidence = conf
    last_matched_question = matched_q
    return RedirectResponse("/", status_code=303)

if __name__ == "__main__":
    uvicorn.run("web_terminal_chat:app", host="127.0.0.1", port=8000, reload=True) 