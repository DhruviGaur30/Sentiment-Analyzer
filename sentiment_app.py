import streamlit as st
import pandas as pd
import numpy as np
import time
import re
from typing import Dict, Tuple

# Streamlit Cloud optimized - lightweight approach
STREAMLIT_CLOUD_OPTIMIZED = True  # Set to False if you want to try AI models locally

if not STREAMLIT_CLOUD_OPTIMIZED:
    # Optional transformers import - only if not optimizing for Streamlit Cloud
    try:
        from transformers import pipeline
        import torch
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
else:
    # Skip AI libraries for Streamlit Cloud optimization
    TRANSFORMERS_AVAILABLE = False

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
        color: #666666;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 40px;
    }
    
    .stTextArea textarea {
        border: 2px solid #e0e0e0 !important;
        border-radius: 15px !important;
        font-size: 1.1rem !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
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
    
    .optimization-badge {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        padding: 8px 16px !important;
        border-radius: 15px !important;
        font-size: 0.8rem !important;
        text-align: center !important;
        margin: 10px auto !important;
        max-width: 300px !important;
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

@st.cache_resource
def load_sentiment_pipeline():
    """Load the sentiment analysis pipeline"""
    if STREAMLIT_CLOUD_OPTIMIZED:
        # Show optimization badge
        st.markdown("""
        <div class="optimization-badge">
            🚀 Optimized for Streamlit Cloud - Lightning Fast!
        </div>
        """, unsafe_allow_html=True)
        return None
        
    if not TRANSFORMERS_AVAILABLE:
        st.info("ℹ️ Using lightweight keyword-based sentiment analysis")
        return None
        
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        return sentiment_pipeline
    except Exception as e:
        st.warning(f"⚠️ AI model unavailable. Using keyword-based analysis.")
        return None

def predict_sentiment(text: str) -> Dict[str, float]:
    """
    Predict sentiment using the loaded pipeline
    """
    pipeline_model = load_sentiment_pipeline()
    
    if pipeline_model is None:
        # Fallback to enhanced keyword analysis
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
    Enhanced keyword-based sentiment analysis - optimized for accuracy
    """
    # Expanded and improved keyword lists
    positive_words = [
        'love', 'like', 'great', 'amazing', 'excellent', 'wonderful', 'fantastic', 
        'awesome', 'brilliant', 'perfect', 'incredible', 'outstanding', 'superb', 
        'marvelous', 'optimistic', 'happy', 'joyful', 'joy', 'excited', 'pleased', 
        'delighted', 'satisfied', 'impressive', 'nice', 'beautiful', 'charming', 
        'fabulous', 'adorable', 'cool', 'lovely', 'appreciate', 'grateful', 
        'commendable', 'terrific', 'breathtaking', 'genius', 'enjoyed', 'positive', 
        'fun', 'rewarding', 'worthwhile', 'valuable', 'helpful', 'supportive', 
        'glad', 'win', 'winning', 'peaceful', 'successful', 'blessed', 'thrilled',
        'elated', 'ecstatic', 'phenomenal', 'spectacular', 'magnificent', 'divine',
        'splendid', 'glorious', 'triumphant', 'victorious', 'blissful', 'euphoric',
        'appeal'  # moved here as it's generally positive
    ]

    negative_words = [
        'hate', 'dislike', 'terrible', 'awful', 'bad', 'horrible', 'disgusting', 
        'worst', 'disappointing', 'disappointed', 'sad', 'angry', 'frustrated', 
        'annoying', 'annoyed', 'pathetic', 'useless', 'stupid', 'dumb', 'lame', 
        'gross', 'nasty', 'boring', 'poor', 'mediocre', 'painful', 'tragic', 'depressing', 
        'miserable', 'unhappy', 'waste', 'hated', 'trash', 'garbage', 
        'crap', 'problematic', 'broken', 'defective', 'buggy', 'slow', 'fail', 
        'failed', 'failure', 'ridiculous', 'nonsense', 'meaningless', 
        'toxic', 'fake', 'cheated', 'scam', 'unreliable', 'catastrophic',
        'disastrous', 'horrendous', 'atrocious', 'appalling', 'abysmal', 'deplorable',
        'contemptible', 'despicable', 'detestable', 'loathsome', 'repulsive', 'revolting',
        'overhyped'  # added this negative word
    ]

    # Neutral/context words that shouldn't count as positive
    neutral_context_words = ['feel like', 'looks like', 'seems like', 'sounds like']
    
    # Intensity modifiers
    intensifiers = ['very', 'extremely', 'really', 'absolutely', 'completely', 'totally', 'quite', 'rather', 'so', 'too']
    diminishers = ['slightly', 'somewhat', 'a bit', 'kind of', 'sort of', 'not very', 'barely', 'hardly', 'maybe']
    
    # Enhanced negation detection
    negation_words = ['not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 
                     'cannot', "can't", "won't", "don't", "doesn't", "isn't", "aren't", 
                     "wasn't", "weren't", "didn't", "haven't", "hasn't", "hadn't", "wouldn't", "shouldn't", "couldn't"]
    
    # Process text
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Check for neutral context phrases first
    for neutral_phrase in neutral_context_words:
        if neutral_phrase in text_lower:
            text_lower = text_lower.replace(neutral_phrase, '')
    
    positive_score = 0
    negative_score = 0
    
    # Enhanced negation detection - look in a wider window
    negated_indices = set()
    for i, word in enumerate(words):
        if word in negation_words:
            # Mark next 5 words as potentially negated
            for j in range(i+1, min(i+6, len(words))):
                negated_indices.add(j)
    
    # Process each word
    for i, word in enumerate(words):
        # Check if this word is negated
        negated = i in negated_indices
        
        # Check for intensifiers
        intensity = 1.0
        if i > 0 and words[i-1] in intensifiers:
            intensity = 1.5
        elif i > 0 and words[i-1] in diminishers:
            intensity = 0.5
            
        # Score positive words
        if word in positive_words:
            score = intensity
            if negated:
                negative_score += score * 1.2  # Negated positive words are more negative
            else:
                positive_score += score
                
        # Score negative words  
        if word in negative_words:
            score = intensity
            if negated:
                positive_score += score * 0.8  # Negated negative words are less positive
            else:
                negative_score += score
    
    # Additional context analysis
    # Look for phrases that indicate dissatisfaction
    dissatisfaction_phrases = [
        "didn't understand", "don't understand", "didn't like", "don't like",
        "wasted my time", "waste of time", "not worth", "overhyped",
        "didn't enjoy", "don't enjoy"
    ]
    
    for phrase in dissatisfaction_phrases:
        if phrase in text_lower:
            negative_score += 2.0
    
    # Look for phrases that indicate satisfaction
    satisfaction_phrases = [
        "loved it", "really enjoyed", "highly recommend", "worth it",
        "exceeded expectations", "amazing experience"
    ]
    
    for phrase in satisfaction_phrases:
        if phrase in text_lower:
            positive_score += 2.0
    
    # Determine sentiment with improved logic
    total_score = positive_score + negative_score
    
    if total_score == 0:
        sentiment = 'neutral'
        confidence = 0.5
    else:
        score_diff = abs(positive_score - negative_score)
        
        if positive_score > negative_score:
            sentiment = 'positive'
            confidence = min(0.95, 0.55 + (score_diff / total_score) * 0.4)
        elif negative_score > positive_score:
            sentiment = 'negative'
            confidence = min(0.95, 0.55 + (score_diff / total_score) * 0.4)
        else:
            sentiment = 'neutral'
            confidence = 0.5
            
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
    st.markdown('<h1 class="title">🤗 AI Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover the emotional tone of any text with lightning-fast analysis</p>', unsafe_allow_html=True)
    
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
            time.sleep(0.5)  # Reduced delay for faster UX
            
            # Get sentiment analysis result
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
        "I am feeling quite optimistic about the future despite recent challenges.",
        "This is the worst product I've ever bought. Completely useless and poorly made.",
        "Amazing work! You've exceeded all my expectations. Truly outstanding performance!"
    ]
    
    for i, example in enumerate(examples):
        if st.button(f'"{example[:50]}..."', key=f"example_{i}", help="Click to use this example"):
            st.session_state.text_input = example
            st.rerun()

if __name__ == "__main__":
    main()