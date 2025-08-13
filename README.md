# Medicine ML Chatbot

A medical question-answer chatbot based on Semantic Search with Sentence Transformers.

## 🚀 Live Demo
[Deployment-Link here - Add after Vercel deployment]

## ✨ Features
- **Semantic Search** for medical questions
- **Professional Medical UI** with clinical design
- **Real-time answers** with Confidence-Scores
- **MedQuad-Dataset** Integration
- **FastAPI Backend** with modern Web-API

## 🛠️ Tech Stack
- **Backend:** FastAPI, Uvicorn
- **ML:** SentenceTransformers, scikit-learn
- **Frontend:** Jinja2 Templates
- **Deployment:** Vercel

## 📦 Installation & Setup

### Local Development
```bash
# Clone repository
git clone [your-repo-url]
cd medicine-ml-chatbot-1

# Install dependencies
pip install -r requirements.txt

# Start application
python web_terminal_chat.py
```

The application will run at: `http://localhost:8000`

### Vercel Deployment
1. **Connect repository to Vercel**
2. **Build Settings:**
   - Framework Preset: `Other`
   - Build Command: `pip install -r requirements.txt`
   - Output Directory: `.`
   - Install Command: `pip install -r requirements.txt`

3. **Environment Variables** (if needed):
   - No special configuration required

## 📊 Dataset
The project uses the **MedQuad-Dataset** with medical FAQ pairs:
- **Size:** ~22MB
- **Format:** JSON with question-answer pairs
- **Source:** Medical literature

## 🔧 API Endpoints
- `GET /` - Main page with chat interface
- `POST /chat` - Chat endpoint for questions
- `GET /health` - Health check endpoint

## ⚠️ Disclaimer
**This is a demonstration system for educational purposes. For medical advice, please consult a qualified healthcare professional.**

## 🎯 Use Cases
- **Company Presentations** - ML/Data Science Demo
- **Prototyping** - Semantic Search Applications
- **Learning Project** - FastAPI + ML Integration

## 📈 Performance
- **Latency:** ~100-500ms per request
- **Model:** all-MiniLM-L6-v2 (384-dimensional)
- **Similarity:** Cosine Similarity

## 💡 Example Questions
- "What are the symptoms of diabetes?"
- "How to treat high blood pressure?"
- "What causes chest pain?"
- "Side effects of antibiotics"
- "How to prevent heart disease?"

## 🔮 Next Steps
- [ ] Model fine-tuning
- [ ] Enhanced UI features
- [ ] Performance optimization
- [ ] Extended datasets