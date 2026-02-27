# AI-Powered Chatbot

## Industrial Project - Codec Technologies

### Project Overview
An intelligent chatbot built using Natural Language Processing (NLP) for customer support and FAQs. The chatbot uses NLTK for text processing and provides contextual responses based on user intent.

### Tech Stack
- **Python** - Core programming language
- **NLTK** - Natural Language Toolkit for text processing
- **Flask** - Web framework for the API and UI
- **SQLite** - Database for conversation logging

### Core Concepts
- Natural Language Processing (NLP)
- Intent Classification
- Text Preprocessing (Tokenization, Lemmatization)
- Conversation Logging & Analytics

### Features
- ✅ Intent-based response generation
- ✅ Text preprocessing with NLTK
- ✅ Conversation history logging
- ✅ Quick reply suggestions
- ✅ Analytics dashboard
- ✅ Beautiful responsive UI
- ✅ Session management

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (automatic on first run):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

### API Endpoints
- `GET /` - Main chatbot interface
- `POST /chat` - Send message and get response
  - Body: `{"message": "your message", "session_id": "optional"}`
- `GET /history/<session_id>` - Get conversation history
- `GET /analytics` - Get chatbot analytics

### Supported Intents
- **Greetings** - Hello, Hi, Hey, etc.
- **Goodbye** - Bye, See you, etc.
- **Thanks** - Thank you, Appreciate, etc.
- **About** - Who are you, What are you, etc.
- **Help** - Help, Support, Assist, etc.
- **Services** - What services, Offerings, etc.
- **Pricing** - Price, Cost, How much, etc.
- **Contact** - Contact, Email, Phone, etc.
- **Hours** - Working hours, Available, etc.
- **Technical** - Issue, Problem, Error, etc.

### Project Structure
```
industrial Project/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── chatbot.db          # SQLite database (created on first run)
```

### Extending the Chatbot
To add new intents, modify the `KNOWLEDGE_BASE` dictionary in `app.py`:

```python
"new_intent": {
    "patterns": ["keyword1", "keyword2", "phrase"],
    "responses": [
        "Response 1",
        "Response 2"
    ]
}
```

### Outcome
- Deploy a chatbot with contextual responses
- User interaction logs for analysis
- Foundation for more advanced NLP implementations

---
**Author:** Codec Technologies Internship Program
**Date:** February 2026
