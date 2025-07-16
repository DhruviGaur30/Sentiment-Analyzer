import streamlit as st
import pandas as pd
import numpy as np
import time
import re
from typing import Dict, Tuple
from transformers import pipeline
import torch

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match the original design
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom styling */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #390F67 100%);
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 40px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        margin: 20px auto;
        max-width: 600px;
    }
    
    .title {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 15px !important;
    }
    
    .subtitle {
        color: #ffffff;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 40px;
    }
    
    .stTextArea > div > div > textarea {
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        font-size: 1.1rem;
        padding: 20px;
        background: rgba(0, 0, 0, 0.9);
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.2) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        padding: 15px 40px !important;
        border-radius: 25px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4) !important;
    }
    
    .sentiment-positive {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        color: white !important;
        padding: 25px !important;
        border-radius: 20px !important;
        text-align: center !important;
        box-shadow: 0 15px 30px rgba(79, 172, 254, 0.3) !important;
        margin: 15px 0 !important;
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%) !important;
        color: white !important;
        padding: 25px !important;
        border-radius: 20px !important;
        text-align: center !important;
        box-shadow: 0 15px 30px rgba(250, 112, 154, 0.3) !important;
        margin: 15px 0 !important;
    }
    
    .sentiment-neutral {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%) !important;
        color: #333 !important;
        padding: 25px !important;
        border-radius: 20px !important;
        text-align: center !important;
        box-shadow: 0 15px 30px rgba(168, 237, 234, 0.3) !important;
        margin: 15px 0 !important;
    }
    
    .sentiment-emoji {
        font-size: 2rem !important;
        margin-bottom: 10px !important;
        display: block !important;
    }
    
    .sentiment-text {
        font-weight: 600 !important;
        font-size: 1.3rem !important;
        margin-bottom: 15px !important;
    }
    
    .confidence-container {
        margin-top: 15px !important;
    }
    
    .confidence-bar {
        background: rgba(255, 255, 255, 0.3) !important;
        height: 8px !important;
        border-radius: 4px !important;
        overflow: hidden !important;
        margin: 10px 0 !important;
    }
    
    .confidence-text {
        font-size: 0.9rem !important;
        opacity: 0.9 !important;
    }
    
    .example-container {
        margin-top: 30px !important;
        padding: 20px !important;
        background: rgba(102, 126, 234, 0.05) !important;
        border-radius: 15px !important;
    }
    
    .example-title {
        color: #667eea !important;
        font-weight: 600 !important;
        margin-bottom: 15px !important;
        text-align: center !important;
    }
    
    .example-text {
        background: rgba(102, 126, 234, 0.1) !important;
        padding: 12px 15px !important;
        border-radius: 10px !important;
        margin: 8px 0 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        font-size: 0.9rem !important;
        color: #555 !important;
        border: none !important;
        width: 100% !important;
    }
    
    .example-text:hover {
        background: rgba(102, 126, 234, 0.2) !important;
        transform: translateX(5px) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-container {
            padding: 30px 20px !important;
            margin: 10px !important;
        }
        
        .title {
            font-size: 2rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load the sentiment analysis pipeline (using the same model as your notebook)
@st.cache_resource
def load_sentiment_pipeline():
    """Load the sentiment analysis pipeline"""
    try:
        # This uses the same distilbert model as shown in your notebook
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        return sentiment_pipeline
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_sentiment(text: str) -> Dict[str, float]:
    """
    Predict sentiment using the loaded pipeline
    """
    pipeline_model = load_sentiment_pipeline()
    
    if pipeline_model is None:
        # Fallback to simple analysis if model fails to load
        return simulate_sentiment_analysis(text)
    
    try:
        # Get prediction from the model
        result = pipeline_model(text)[0]
        
        # Map the model output to our format
        label = result['label'].upper()
        score = result['score']
        
        # Convert labels to our format
        if label == 'POSITIVE':
            sentiment = 'positive'
        elif label == 'NEGATIVE':
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'sentiment': sentiment,
            'confidence': score
        }
        
    except Exception as e:
        st.error(f"Error predicting sentiment: {str(e)}")
        return simulate_sentiment_analysis(text)

def simulate_sentiment_analysis(text: str) -> Dict[str, float]:
    """
    Fallback sentiment analysis if model fails
    """
    positive_words = ['love', 'great', 'amazing', 'excellent', 'wonderful', 'fantastic', 
                     'awesome', 'brilliant', 'perfect', 'incredible', 'outstanding', 
                     'superb', 'marvelous', 'optimistic', 'happy', 'joy', 'excited', 'liked']
    
    negative_words = ['hate', 'terrible', 'awful', 'bad', 'horrible', 'disgusting', 
                     'worst', 'disappointing', 'sad', 'angry', 'frustrated', 'annoying', 
                     'pathetic', 'useless', 'stupid']
    
    words = re.findall(r'\b\w+\b', text.lower())
    positive_count = sum(1 for word in words if any(pw in word for pw in positive_words))
    negative_count = sum(1 for word in words if any(nw in word for nw in negative_words))
    
    if positive_count > negative_count:
        sentiment = 'positive'
        confidence = min(0.95, 0.6 + (positive_count - negative_count) * 0.1)
    elif negative_count > positive_count:
        sentiment = 'negative'  
        confidence = min(0.95, 0.6 + (negative_count - positive_count) * 0.1)
    else:
        sentiment = 'neutral'
        confidence = 0.5 + np.random.random() * 0.3
        
    return {
        'sentiment': sentiment,
        'confidence': confidence
    }

def display_sentiment_result(result: Dict[str, float]):
    """Display the sentiment analysis result with beautiful styling"""
    
    sentiment_config = {
        'positive': {
            'emoji': '😊',
            'text': 'Positive Sentiment',
            'class': 'sentiment-positive'
        },
        'negative': {
            'emoji': '😔', 
            'text': 'Negative Sentiment',
            'class': 'sentiment-negative'
        },
        'neutral': {
            'emoji': '😐',
            'text': 'Neutral Sentiment', 
            'class': 'sentiment-neutral'
        }
    }
    
    config = sentiment_config[result['sentiment']]
    confidence_percent = result['confidence'] * 100
    
    # Create the sentiment card HTML
    sentiment_html = f"""
    <div class="{config['class']}">
        <div class="sentiment-emoji">{config['emoji']}</div>
        <div class="sentiment-text">{config['text']}</div>
        <div class="confidence-container">
            <div class="confidence-bar">
                <div style="
                    height: 100%; 
                    background: rgba(255, 255, 255, 0.8); 
                    border-radius: 4px; 
                    width: {confidence_percent}%;
                    transition: width 1s ease;
                "></div>
            </div>
            <div class="confidence-text">Confidence: {confidence_percent:.1f}%</div>
        </div>
    </div>
    """
    
    st.markdown(sentiment_html, unsafe_allow_html=True)

def main():
    # Title and subtitle
    st.markdown('<h1 class="title"> AI Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover the emotional tone of any text with advanced AI</p>', unsafe_allow_html=True)
    
    # Text input
    user_text = st.text_area(
        "",
        placeholder="Type or paste your text here... Express yourself and let AI understand your sentiment!",
        height=120,
        key="text_input"
    )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button("✨ Analyze Sentiment", key="analyze_btn")
    
    # Process analysis
    if analyze_clicked and user_text.strip():
        # Show loading spinner
        with st.spinner('🔄 Analyzing sentiment...'):
            time.sleep(1)  # Brief delay for UX
            
            # Get sentiment analysis result using your model
            result = predict_sentiment(user_text)
            
        # Display result
        st.markdown("---")
        display_sentiment_result(result)
        
    elif analyze_clicked and not user_text.strip():
        st.error("⚠️ Please enter some text to analyze!")
    
    # Example texts section
    st.markdown("---")
    st.markdown("""
    <div class="example-container">
        <div class="example-title">💡 Try these examples:</div>
    </div>
    """, unsafe_allow_html=True)
    
    examples = [
        "I absolutely love this new technology! It's incredible how AI can understand human emotions.",
        "This movie was terrible. I wasted my time and money on this disappointing experience.", 
        "The weather today is okay. Nothing special, just another ordinary day.",
        "I am feeling quite optimistic about the future despite recent challenges."
    ]
    
    for i, example in enumerate(examples):
        if st.button(f'"{example[:50]}..."', key=f"example_{i}", help="Click to use this example"):
            st.session_state.text_input = example
            st.rerun()

if __name__ == "__main__":
    main()