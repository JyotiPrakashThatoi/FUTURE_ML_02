import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re

# Page config
st.set_page_config(page_title="Support AI | Ticket System", page_icon="🎫", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
        color: #333333; /* Ensure dark text on white background */
    }
    .prediction-card p {
        color: #333333 !important;
    }
    .cat-label {
        font-size: 1.2em;
        font-weight: bold;
        color: #007bff !important;
    }
    .prio-label {
        font-size: 1.2em;
        font-weight: bold;
        color: #dc3545 !important;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #333333 !important;
    }
    .metric-box h4, .metric-box h2 {
        color: #333333 !important;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Load NLP assets
@st.cache_resource
def load_assets():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    model_type = joblib.load('model_type.pkl')
    model_prio = joblib.load('model_prio.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    le_type = joblib.load('le_type.pkl')
    le_prio = joblib.load('le_prio.pkl')
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    return model_type, model_prio, tfidf, le_type, le_prio, lemmatizer, stop_words

# Preprocessing
def clean_text(text, lemmatizer, stop_words):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Content Moderation (Safety Layer)
def is_content_safe(text):
    blocked_keywords = ['sex', 'porn', 'scam', 'gamble'] 
    text_clean = text.lower().strip()
    for word in blocked_keywords:
        if word in text_clean:
            return False
    return True

# Hybrid Prediction Logic (ML + Business Rules)
def get_hybrid_prediction(subject, description, model_type, model_prio, tfidf, le_type, le_prio, lemmatizer, stop_words):
    text = (str(subject) + " " + str(description)).lower()
    
    # 1. High-Confidence Keyword Mappings (Business Rules)
    technical_keywords = ['login', 'password', 'credentials', 'reset', 'error', 'bug', 'crash', 'frozen', 'update', 'installation']
    refund_keywords = ['refund', 'money back', 'return', 'cancel order', 'wrong item']
    billing_keywords = ['billing', 'charge', 'invoice', 'credit card', 'payment', 'subscription', 'pricing']
    cancellation_keywords = ['cancel account', 'terminate', 'close account', 'membership', 'unsubscribe']

    res_type = None
    if any(k in text for k in technical_keywords):
        res_type = "Technical issue"
    elif any(k in text for k in refund_keywords):
        res_type = "Refund request"
    elif any(k in text for k in billing_keywords):
        res_type = "Billing inquiry"
    elif any(k in text for k in cancellation_keywords):
        res_type = "Cancellation request"

    res_prio = "Medium"
    if res_type == "Technical issue":
        res_prio = "High" if any(k in text for k in ['login', 'password', 'crash', 'critical']) else "Medium"
    elif res_type == "Refund request":
        res_prio = "High" if "double charge" in text else "Medium"

    # 2. ML Fallback
    cleaned = clean_text(text, lemmatizer, stop_words)
    vec = tfidf.transform([cleaned])
    
    ml_type = le_type.inverse_transform(model_type.predict(vec))[0]
    ml_prio = le_prio.inverse_transform(model_prio.predict(vec))[0]

    final_type = res_type if res_type else ml_type
    final_prio = res_prio if res_type else ml_prio

    return final_type, final_prio

# Sidebar Navigation
st.sidebar.title("🎫 Support Portal")
page = st.sidebar.radio("Navigation", ["Submit Ticket", "Admin Dashboard"])

if page == "Submit Ticket":
    st.title("🎫 Support Ticket AI Classifier")
    st.markdown("Automate your support workflow with Machine Learning. Enter the ticket details below to classify and prioritize.")

    with st.container():
        subject = st.text_input("Ticket Subject", placeholder="e.g., Cannot access account")
        description = st.text_area("Ticket Description", placeholder="Describe the issue in detail...", height=150)
        
        if st.button("Classify Ticket"):
            if subject and description:
                full_text = f"{subject} {description}"
                if not is_content_safe(full_text):
                    st.error("⚠️ **Safety Warning**: This ticket contains content that violates our professional conduct policy.")
                else:
                    with st.spinner("Analyzing ticket..."):
                        model_type, model_prio, tfidf, le_type, le_prio, lemmatizer, stop_words = load_assets()
                        res_type, res_prio = get_hybrid_prediction(
                            subject, description, model_type, model_prio, tfidf, le_type, le_prio, lemmatizer, stop_words
                        )
                        
                        st.markdown(f"""
                        <div class="prediction-card">
                            <p><strong>Predicted Category:</strong> <span class="cat-label">{res_type}</span></p>
                            <p><strong>Recommended Priority:</strong> <span class="prio-label">{res_prio}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
            else:
                st.warning("Please fill in both fields.")

else: # Admin Dashboard
    st.title("📊 Support Manager Dashboard")
    st.markdown("Review and manage classified tickets across the organization.")

    # Load data for dashboard
    @st.cache_data
    def load_data():
        return pd.read_csv('customer_support_tickets.csv').head(100) # Load subset for demo

    data = load_data()
    
    # Dashboard Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-box"><h4>Total New Tickets</h4><h2>{len(data)}</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><h4>Avg Complexity</h4><h2>Medium</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><h4>System Uptime</h4><h2>99.9%</h2></div>', unsafe_allow_html=True)

    st.markdown("### 📋 Classified Ticket Queue")
    
    num_tickets = st.slider("Number of tickets to display", 5, 50, 10)
    
    if st.button("Automate Classification for Next Batch"):
        with st.spinner("Batch processing..."):
            model_type, model_prio, tfidf, le_type, le_prio, lemmatizer, stop_words = load_assets()
            
            sample = data.head(num_tickets).copy()
            results = []
            for idx, row in sample.iterrows():
                p_type, p_prio = get_hybrid_prediction(
                    row['Ticket Subject'], row['Ticket Description'], 
                    model_type, model_prio, tfidf, le_type, le_prio, lemmatizer, stop_words
                )
                results.append({'Subject': row['Ticket Subject'], 'AI Category': p_type, 'AI Priority': p_prio})
            
            results_df = pd.DataFrame(results)
            st.table(results_df)
            st.success(f"Successfully processed {num_tickets} tickets with AI intelligence.")
    else:
        st.info("Click the button above to view the latest AI classifications for the ticket queue.")

st.markdown("---")
st.caption("Developed for Future Interns | Machine Learning Task 2")
