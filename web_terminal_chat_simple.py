import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
import json
import logging
import random

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medicine ML Chatbot",
    description="A medical question-answer chatbot demo",
    version="1.0.0"
)

# Templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Chat history in memory (for demo purposes)
chat_history = []
last_confidence = None
last_matched_question = None

# Global variables for data
questions = []
answers = []

def load_data():
    """Loads the data on first call"""
    global questions, answers
    
    try:
        logger.info("Loading data...")
        
        # Try to load smaller dataset first (for Vercel), fallback to full dataset
        data_path = os.path.join(BASE_DIR, 'medquad_small.json')
        if not os.path.exists(data_path):
            data_path = os.path.join(BASE_DIR, 'medquad_full.json')
        
        with open(data_path, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
        
        questions = [item['question'] for item in faq_data]
        answers = [item['answer'] for item in faq_data]
        
        logger.info(f"Loaded {len(questions)} questions and answers from {os.path.basename(data_path)}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return False

def retrieve_answer(user_question):
    """Finds a relevant answer for a user question (simplified version)"""
    try:
        if not user_question or len(user_question.strip()) == 0:
            return "Please enter a question.", 0.0, ""
        
        # Simple keyword matching for demo
        user_lower = user_question.lower()
        
        # Find questions with similar keywords
        best_match = None
        best_score = 0.0
        
        for i, question in enumerate(questions):
            question_lower = question.lower()
            
            # Simple keyword matching
            common_words = set(user_lower.split()) & set(question_lower.split())
            if len(common_words) > 0:
                score = len(common_words) / max(len(user_lower.split()), len(question_lower.split()))
                if score > best_score:
                    best_score = score
                    best_match = i
        
        if best_match is not None:
            return answers[best_match], best_score, questions[best_match]
        else:
            # Return a random answer if no match found
            random_idx = random.randint(0, len(answers) - 1)
            return answers[random_idx], 0.1, questions[random_idx]
        
    except Exception as e:
        logger.error(f"Error in retrieve_answer: {e}")
        return "Sorry, there was an error processing your question.", 0.0, ""

@app.on_event("startup")
async def startup_event():
    """Executed when the application starts"""
    success = load_data()
    if not success:
        logger.error("Failed to load data")

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
    return {"status": "healthy", "data_loaded": len(questions) > 0}

if __name__ == "__main__":
    uvicorn.run("web_terminal_chat_simple:app", host="127.0.0.1", port=8000, reload=True)
