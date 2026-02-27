"""
AI-Powered Chatbot
Industrial Project - Codec Technologies

Tech Stack: Python, NLTK, Transformers, Flask/FastAPI, SQLite
Description: Build an intelligent chatbot using Natural Language Processing (NLP)
             for customer support or FAQs.
"""

from flask import Flask, render_template, request, jsonify
import sqlite3
import json
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

app = Flask(__name__)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Knowledge base for the chatbot
KNOWLEDGE_BASE = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"],
        "responses": [
            "Hello! How can I help you today?",
            "Hi there! What can I assist you with?",
            "Hey! Welcome to our support. How may I help you?",
            "Greetings! I'm here to help. What do you need?"
        ]
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you", "take care", "exit", "quit"],
        "responses": [
            "Goodbye! Have a great day!",
            "See you later! Feel free to come back anytime.",
            "Take care! It was nice helping you.",
            "Bye! Don't hesitate to reach out if you need more help."
        ]
    },
    "thanks": {
        "patterns": ["thank", "thanks", "thank you", "appreciate", "grateful"],
        "responses": [
            "You're welcome! Is there anything else I can help with?",
            "Happy to help! Let me know if you need anything else.",
            "My pleasure! Feel free to ask more questions.",
            "Anytime! I'm here to assist you."
        ]
    },
    "about": {
        "patterns": ["who are you", "what are you", "about", "introduce", "your name"],
        "responses": [
            "I'm an AI-powered chatbot designed to help answer your questions and provide support.",
            "I'm your virtual assistant, here to help with any queries you might have.",
            "I'm an intelligent chatbot built using Natural Language Processing to assist you."
        ]
    },
    "help": {
        "patterns": ["help", "support", "assist", "guide", "how to"],
        "responses": [
            "I can help you with various topics! Try asking about our services, products, pricing, or technical support.",
            "I'm here to assist! You can ask me about FAQs, services, or any general questions.",
            "Need help? I can answer questions about our company, products, services, and more!"
        ]
    },
    "services": {
        "patterns": ["service", "services", "what do you offer", "offerings", "provide"],
        "responses": [
            "We offer a wide range of services including software development, AI solutions, cloud computing, and technical consulting.",
            "Our services include: 1) Custom Software Development, 2) AI & ML Solutions, 3) Cloud Services, 4) Technical Support.",
            "We provide comprehensive IT solutions tailored to your business needs."
        ]
    },
    "pricing": {
        "patterns": ["price", "pricing", "cost", "how much", "fee", "charge"],
        "responses": [
            "Our pricing varies based on your specific requirements. Please contact our sales team for a customized quote.",
            "We offer competitive pricing! For detailed pricing information, please reach out to sales@company.com.",
            "Pricing depends on the scope of your project. Would you like me to connect you with our sales team?"
        ]
    },
    "contact": {
        "patterns": ["contact", "reach", "email", "phone", "call"],
        "responses": [
            "You can reach us at: Email: support@company.com, Phone: +1-800-123-4567",
            "Contact us via email at support@company.com or call us at +1-800-123-4567.",
            "Our support team is available 24/7. Email: support@company.com"
        ]
    },
    "hours": {
        "patterns": ["hours", "timing", "open", "available", "working hours"],
        "responses": [
            "Our support is available 24/7! Feel free to reach out anytime.",
            "We're here for you round the clock - 24 hours a day, 7 days a week.",
            "Our team is available 24/7 to assist you with any queries."
        ]
    },
    "technical": {
        "patterns": ["technical", "issue", "problem", "error", "bug", "not working"],
        "responses": [
            "I'm sorry to hear you're experiencing technical issues. Can you please describe the problem in detail?",
            "Let me help you with that technical issue. Please provide more details about what's happening.",
            "Technical problems can be frustrating. Please share the error message or describe the issue, and I'll try to help."
        ]
    },
    "default": {
        "patterns": [],
        "responses": [
            "I'm not sure I understand. Could you please rephrase your question?",
            "I didn't quite catch that. Can you try asking in a different way?",
            "I'm still learning! Could you please provide more details or ask differently?",
            "Hmm, I'm not sure about that. Would you like to speak with a human agent?"
        ]
    }
}

# Database setup
def init_db():
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            intent TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            rating INTEGER,
            comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def preprocess_text(text):
    """Preprocess text for NLP analysis"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens

def calculate_similarity(user_tokens, pattern_tokens):
    """Calculate similarity between user input and pattern"""
    if not user_tokens or not pattern_tokens:
        return 0
    
    user_set = set(user_tokens)
    pattern_set = set(pattern_tokens)
    
    intersection = user_set.intersection(pattern_set)
    union = user_set.union(pattern_set)
    
    if not union:
        return 0
    
    return len(intersection) / len(union)

def get_intent(user_message):
    """Determine the intent of the user message"""
    user_tokens = preprocess_text(user_message)
    
    best_intent = "default"
    best_score = 0
    
    for intent, data in KNOWLEDGE_BASE.items():
        if intent == "default":
            continue
            
        for pattern in data["patterns"]:
            pattern_tokens = preprocess_text(pattern)
            score = calculate_similarity(user_tokens, pattern_tokens)
            
            # Also check for exact word matches
            for token in user_tokens:
                if token in pattern.lower():
                    score += 0.3
            
            if score > best_score:
                best_score = score
                best_intent = intent
    
    # Threshold for intent detection
    if best_score < 0.2:
        best_intent = "default"
        best_score = 0
    
    return best_intent, best_score

def get_response(user_message, session_id="default"):
    """Generate a response for the user message"""
    intent, confidence = get_intent(user_message)
    response = random.choice(KNOWLEDGE_BASE[intent]["responses"])
    
    # Log conversation to database
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO conversations (session_id, user_message, bot_response, intent, confidence) VALUES (?, ?, ?, ?, ?)',
        (session_id, user_message, response, intent, confidence)
    )
    conn.commit()
    conn.close()
    
    return {
        "response": response,
        "intent": intent,
        "confidence": round(confidence, 2)
    }

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Powered Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .chat-container {
            width: 100%;
            max-width: 500px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }
        .chat-header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        .chat-header p {
            opacity: 0.8;
            font-size: 14px;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #4CAF50;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .chat-messages {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        .message.user {
            flex-direction: row-reverse;
        }
        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            flex-shrink: 0;
        }
        .message.bot .message-avatar {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            margin-right: 10px;
        }
        .message.user .message-avatar {
            background: #e0e0e0;
            margin-left: 10px;
        }
        .message-content {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 18px;
            font-size: 14px;
            line-height: 1.5;
        }
        .message.bot .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border-bottom-right-radius: 5px;
        }
        .chat-input {
            display: flex;
            padding: 20px;
            background: white;
            border-top: 1px solid #eee;
        }
        .chat-input input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 30px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }
        .chat-input input:focus {
            border-color: #1e3c72;
        }
        .chat-input button {
            margin-left: 10px;
            padding: 15px 25px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .chat-input button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(30, 60, 114, 0.4);
        }
        .typing-indicator {
            display: none;
            padding: 10px 20px;
            color: #666;
            font-style: italic;
        }
        .quick-replies {
            padding: 10px 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .quick-reply {
            padding: 8px 15px;
            background: #e8f0fe;
            color: #1e3c72;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 12px;
            transition: background 0.2s;
        }
        .quick-reply:hover {
            background: #d0e0fc;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🤖 AI Chatbot</h1>
            <p><span class="status-indicator"></span>Online | Powered by NLP</p>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    Hello! I'm your AI assistant. How can I help you today?
                </div>
            </div>
        </div>
        
        <div class="quick-replies">
            <button class="quick-reply" onclick="sendQuickReply('What services do you offer?')">Services</button>
            <button class="quick-reply" onclick="sendQuickReply('What are your pricing options?')">Pricing</button>
            <button class="quick-reply" onclick="sendQuickReply('How can I contact you?')">Contact</button>
            <button class="quick-reply" onclick="sendQuickReply('I need help')">Help</button>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            Bot is typing...
        </div>
        
        <div class="chat-input">
            <input type="text" id="userInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    
    <script>
        const sessionId = 'session_' + Date.now();
        
        function addMessage(content, isUser) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user' : 'bot');
            messageDiv.innerHTML = `
                <div class="message-avatar">${isUser ? '👤' : '🤖'}</div>
                <div class="message-content">${content}</div>
            `;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function showTyping() {
            document.getElementById('typingIndicator').style.display = 'block';
        }
        
        function hideTyping() {
            document.getElementById('typingIndicator').style.display = 'none';
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage(message, true);
            input.value = '';
            
            showTyping();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: message,
                        session_id: sessionId
                    })
                });
                
                const data = await response.json();
                
                setTimeout(() => {
                    hideTyping();
                    addMessage(data.response, false);
                }, 500 + Math.random() * 1000);
                
            } catch (error) {
                hideTyping();
                addMessage('Sorry, I encountered an error. Please try again.', false);
            }
        }
        
        function sendQuickReply(message) {
            document.getElementById('userInput').value = message;
            sendMessage();
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return HTML_TEMPLATE

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    response = get_response(user_message, session_id)
    return jsonify(response)

@app.route('/history/<session_id>')
def get_history(session_id):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT user_message, bot_response, intent, confidence, timestamp FROM conversations WHERE session_id = ? ORDER BY timestamp',
        (session_id,)
    )
    conversations = cursor.fetchall()
    conn.close()
    
    return jsonify({
        "conversations": [
            {
                "user_message": c[0],
                "bot_response": c[1],
                "intent": c[2],
                "confidence": c[3],
                "timestamp": c[4]
            } for c in conversations
        ]
    })

@app.route('/analytics')
def analytics():
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    
    # Get intent distribution
    cursor.execute('SELECT intent, COUNT(*) FROM conversations GROUP BY intent')
    intent_stats = cursor.fetchall()
    
    # Get total conversations
    cursor.execute('SELECT COUNT(*) FROM conversations')
    total = cursor.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        "total_conversations": total,
        "intent_distribution": {i[0]: i[1] for i in intent_stats}
    })

if __name__ == '__main__':
    print("Starting AI-Powered Chatbot...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
