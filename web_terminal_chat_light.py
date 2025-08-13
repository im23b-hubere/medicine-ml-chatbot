import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
import json
import logging
import re
from difflib import SequenceMatcher

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medicine ML Chatbot",
    description="A medical question-answer chatbot using text similarity",
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
        
        # Try to load tiny dataset first (for Vercel)
        data_path = os.path.join(BASE_DIR, 'medquad_tiny.json')
        if not os.path.exists(data_path):
            logger.error("No dataset found")
            return False
        
        with open(data_path, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
        
        questions = [item['question'] for item in faq_data]
        answers = [item['answer'] for item in faq_data]
        
        logger.info(f"Loaded {len(questions)} questions and answers")
        return True
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return False

def text_similarity(text1, text2):
    """Calculate text similarity using SequenceMatcher"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def retrieve_answer(user_question):
    """Finds the best answer using text similarity"""
    try:
        if not user_question or len(user_question.strip()) == 0:
            return "Please enter a question.", 0.0, ""
        
        # Clean user question
        user_clean = re.sub(r'[^\w\s]', '', user_question.lower())
        
        best_match = None
        best_score = 0.0
        
        for i, question in enumerate(questions):
            # Clean question
            question_clean = re.sub(r'[^\w\s]', '', question.lower())
            
            # Calculate similarity
            similarity = text_similarity(user_clean, question_clean)
            
            # Also check for keyword overlap
            user_words = set(user_clean.split())
            question_words = set(question_clean.split())
            if len(user_words) > 0 and len(question_words) > 0:
                keyword_overlap = len(user_words & question_words) / len(user_words | question_words)
                combined_score = (similarity + keyword_overlap) / 2
            else:
                combined_score = similarity
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = i
        
        if best_match is not None and best_score > 0.1:
            return answers[best_match], best_score, questions[best_match]
        else:
            # Return a default answer if no good match
            return "I don't have a specific answer for that question. Please try rephrasing or ask about common medical topics like diabetes, heart disease, or symptoms.", 0.05, "No specific match found"
        
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
    uvicorn.run("web_terminal_chat_light:app", host="127.0.0.1", port=8000, reload=True)
