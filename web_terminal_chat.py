import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medicine ML Chatbot",
    description="A medical question-answer chatbot using Semantic Search",
    version="1.0.0"
)

# Templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Chat history in memory (for demo purposes)
chat_history = []
last_confidence = None
last_matched_question = None

# Global variables for model and data
model = None
question_embeddings = None
questions = []
answers = []

def load_model_and_data():
    """Loads the model and data on first call"""
    global model, question_embeddings, questions, answers
    
    try:
        logger.info("Loading model and data...")
        
        # Try to load tiny dataset first (for Render memory constraints), then larger ones
        data_paths = [
            os.path.join(BASE_DIR, 'medquad_tiny.json'),
            os.path.join(BASE_DIR, 'medquad_small.json'),
            os.path.join(BASE_DIR, 'medquad_full.json')
        ]
        
        data_path = None
        for path in data_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if not data_path:
            logger.error("No dataset found")
            return False
        
        with open(data_path, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
        
        questions = [item['question'] for item in faq_data]
        answers = [item['answer'] for item in faq_data]
        
        # Load model - use standard model for Render (memory constraints)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded standard model for Render deployment")
        question_embeddings = model.encode(questions, show_progress_bar=False, convert_to_numpy=True)
        
        logger.info(f"Loaded {len(questions)} questions and answers from {os.path.basename(data_path)}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def retrieve_answer(user_question):
    """Finds the best answer for a user question"""
    try:
        if not user_question or len(user_question.strip()) == 0:
            return "Please enter a question.", 0.0, ""
        
        user_emb = model.encode([user_question], convert_to_numpy=True)
        sims = cosine_similarity(user_emb, question_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_question = questions[best_idx]
        answer = answers[best_idx]
        
        return answer, best_score, best_question
        
    except Exception as e:
        logger.error(f"Error in retrieve_answer: {e}")
        return "Sorry, there was an error processing your question.", 0.0, ""

@app.on_event("startup")
async def startup_event():
    """Executed when the application starts"""
    success = load_model_and_data()
    if not success:
        logger.error("Failed to load model and data")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Main page with chat interface"""
    try:
        return templates.TemplateResponse("terminal_chat.html", {
            "request": request,
            "chat_history": chat_history,
            "last_confidence": last_confidence,
            "last_matched_question": last_matched_question,
        })
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat", response_class=HTMLResponse)
def chat(request: Request, user_input: str = Form(...)):
    """Chat endpoint for user questions"""
    global last_confidence, last_matched_question
    
    try:
        # Validate input
        if not user_input or len(user_input.strip()) == 0:
            return RedirectResponse("/", status_code=303)
        
        # Generate answer
        answer, conf, matched_q = retrieve_answer(user_input.strip())
        
        # Update chat history (max 10 entries for performance)
        chat_history.append((user_input.strip(), answer, conf, matched_q))
        if len(chat_history) > 10:
            chat_history.pop(0)
        
        last_confidence = conf
        last_matched_question = matched_q
        
        return RedirectResponse("/", status_code=303)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return RedirectResponse("/", status_code=303)

@app.get("/health")
def health_check():
    """Health check endpoint for Vercel"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run("web_terminal_chat:app", host="127.0.0.1", port=8000, reload=True) 