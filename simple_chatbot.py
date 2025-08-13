import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
import json
import logging

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Simple Medical Chatbot",
    description="A simple medical Q&A chatbot",
    version="1.0.0"
)

# Templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Chat history
chat_history = []

# Simple keyword-based matching
def simple_answer(user_question):
    """Simple keyword-based answer system"""
    user_question = user_question.lower()
    
    # Simple keyword matching
    if "diabetes" in user_question:
        return "Diabetes is a chronic condition that affects how your body turns food into energy. There are two main types: Type 1 and Type 2.", 0.8, "What is diabetes?"
    
    elif "blood pressure" in user_question or "hypertension" in user_question:
        return "High blood pressure (hypertension) is a common condition that affects the body's arteries. It's often called the 'silent killer' because it typically has no symptoms.", 0.7, "What is high blood pressure?"
    
    elif "heart" in user_question:
        return "The heart is a muscular organ that pumps blood throughout the body. Common heart conditions include coronary artery disease, heart failure, and arrhythmias.", 0.6, "Heart health information"
    
    elif "cancer" in user_question:
        return "Cancer is a group of diseases characterized by the uncontrolled growth and spread of abnormal cells. Early detection and treatment are crucial.", 0.6, "Cancer information"
    
    elif "pain" in user_question:
        return "Pain is a complex sensation that can be acute or chronic. It's important to consult with healthcare professionals for proper diagnosis and treatment.", 0.5, "Pain management"
    
    else:
        return "I'm a simple medical chatbot. Please ask about specific conditions like diabetes, blood pressure, heart health, cancer, or pain management.", 0.3, "General medical information"

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Main page with chat interface"""
    try:
        return templates.TemplateResponse("terminal_chat.html", {
            "request": request,
            "chat_history": chat_history,
            "last_confidence": None,
            "last_matched_question": None,
        })
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat", response_class=HTMLResponse)
def chat(request: Request, user_input: str = Form(...)):
    """Chat endpoint for user questions"""
    try:
        if not user_input or len(user_input.strip()) == 0:
            return RedirectResponse("/", status_code=303)
        
        # Generate simple answer
        answer, conf, matched_q = simple_answer(user_input.strip())
        
        # Update chat history
        chat_history.append((user_input.strip(), answer, conf, matched_q))
        if len(chat_history) > 10:
            chat_history.pop(0)
        
        return RedirectResponse("/", status_code=303)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return RedirectResponse("/", status_code=303)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "type": "simple_chatbot"}

if __name__ == "__main__":
    uvicorn.run("simple_chatbot:app", host="127.0.0.1", port=8000, reload=True)
