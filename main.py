from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
# Using the new 'ddgs' import as recommended for Method 2
from ddgs import DDGS 
import joblib 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- APPLICATION INSTANCE ---
app = FastAPI(
    title="KhabarGPT Fake News Detector API",
    description="Backend for the Semester 7 Capstone Project. Analyzes news headlines using a three-pronged ensemble voting system.",
    version="1.0.0"
)

# --- CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL TEXT CLEANING FUNCTION (MUST match train_method1.py) ---
# This function must be identical to the one used during TF-IDF training.
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text_method1(text: str) -> str:
        """Applies NLTK cleaning: lowercase, regex, stopword removal, and lemmatization."""
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)
except Exception as e:
    # Error handling for NLTK setup
    print(f"⚠️ NLTK Setup Error: {e}")

# --- 1. LOAD AI MODEL (Method 3: General Style) ---
print("⏳ Loading AI Model (Method 3)...")
try:
    # Model: jy46604790/Fake-News-Bert-Detect for general style/linguistic analysis.
    # Prediction: LABEL_0 = Fake, LABEL_1 = Real.
    ai_classifier = pipeline("text-classification", model="jy46604790/Fake-News-Bert-Detect")
    method3_active = True
except Exception as e:
    print(f"⚠️ AI Model Load Error: {e}")
    method3_active = False

# --- 2. LOAD CUSTOM ML MODEL (Method 1: Indian Context) ---
print("⏳ Loading Custom ML Model (Method 1)...")
try:
    # Load the TF-IDF Vectorizer and Logistic Regression Model
    m1_model = joblib.load('method1_model.pkl')
    m1_vectorizer = joblib.load('method1_vectorizer.pkl')
    method1_active = True
except (FileNotFoundError, AttributeError):
    print("⚠️ Method 1 Failed to Load: Missing .pkl files. Did you run 'train_method1.py'?")
    method1_active = False
except Exception as e:
    print(f"⚠️ Method 1 Loading Error: {e}")
    method1_active = False

# --- HELPER FUNCTIONS ---

def method_1_custom_model(text: str) -> dict:
    """
    METHOD 1: Specialized Fake News Classifier (TF-IDF + Logistic Regression).
    Trained on the local BharatFakeNewsKosh dataset for Indian context predictions.
    """
    if not method1_active:
        return {"prediction": "ERROR", "confidence": 0.0, "details": "Model not loaded"}

    cleaned_text = clean_text_method1(text)
    # Use .transform, NOT .fit_transform
    vectorized_text = m1_vectorizer.transform([cleaned_text])
    
    # Predict the probability [prob_real, prob_fake]
    prediction_prob = m1_model.predict_proba(vectorized_text)[0]
    fake_prob = prediction_prob[1] 
    
    verdict = "FAKE" if fake_prob >= 0.5 else "REAL"
    confidence = fake_prob if verdict == "FAKE" else (1 - fake_prob)
    
    return {
        "prediction": verdict,
        "confidence": round(confidence * 100, 2),
        "details": "Based on BharatKosh training data"
    }

def method_2_web_search(text: str) -> dict:
    """
    METHOD 2: Live Fact-Check Verification (Web Scraping Heuristics).
    Searches trusted fact-checking sites to see if the claim is already debunked.
    """
    try:
        # Using DDGS (DuckDuckGo Search) for fast, free access to search results
        results = DDGS().text(f"{text} fact check", max_results=5)
        if not results:
            return {"flagged": False, "links": []}

        suspicious_keywords = ["fake", "hoax", "false", "debunked", "rumour", "misleading"]
        flagged_links = []
        
        # Trusted sources list, crucial for authoritative debunking
        trusted_domains = [
           "altnews.in", "boomlive.in", "thequint.com/news/webqoof", 
           "pib.gov.in/factcheck", "snopes.com", "factcheck.org"
        ]

        for res in results:
            title = res.get('title', '').lower()
            snippet = res.get('body', '').lower()
            link = res.get('href', '')

            # Check 1: Check for explicit negative keywords
            has_flag_keyword = any(word in title for word in suspicious_keywords) or any(word in snippet for word in suspicious_keywords)

            # Check 2: Check if the result is from an authoritative, trusted source
            is_trusted_source = any(domain in link for domain in trusted_domains)

            # If either condition is met, we flag the result as a debunking link
            if has_flag_keyword or is_trusted_source:
                # We only append ONCE per result to avoid duplication
                flagged_links.append({"title": res.get('title'), "link": link})
            
        return {"flagged": len(flagged_links) > 0, "links": flagged_links}
    
    except Exception as e:
        print(f"Search Error: {e}")
        return {"flagged": False, "links": [], "error": str(e)}

def method_3_ai_prediction(text: str) -> dict:
    """
    METHOD 3: General Linguistic Analysis (BERT Transformer).
    Analyzes the writing style (sentiment, urgency, complexity) to predict FAKE or REAL.
    """
    if not method3_active:
        return {"prediction": "ERROR", "confidence": 0.0, "details": "Model not loaded"}
        
    # Truncate to 512 chars to fit the BERT model's limits
    result = ai_classifier(text[:512])[0]
    
    label = result['label']
    score = result['score']
    
    # Normalize prediction to FAKE/REAL
    prediction = "FAKE" if label == "LABEL_0" else "REAL"
    
    return {"prediction": prediction, "confidence": score}


# --- API INPUT/OUTPUT MODEL (Pydantic) ---

class NewsRequest(BaseModel):
    """Defines the expected format for the incoming JSON request."""
    content: str

@app.post(
    "/analyze",
    summary="Runs the 3-Method Ensemble Prediction System.",
    description="Processes a news headline through the Custom Model, Web Search, and General AI Model to return a final confidence-based verdict.",
)
async def analyze_news(request: NewsRequest) -> dict:
    """
    API ENDPOINT: Executes the core logic of the KhabarGPT project.
    """
    text = request.content
    print(f"\n🔍 Analyzing: {text[:50]}...")

    # --- RUN ALL 3 METHODS CONCURRENTLY ---
    # Note: In production, these should be run with asyncio.gather() for speed.
    m1_res = method_1_custom_model(text)
    m2_res = method_2_web_search(text)
    m3_res = method_3_ai_prediction(text)

    # --- CALCULATE FINAL VERDICT (Consensus Logic) ---
    final_verdict = "UNCERTAIN"
    confidence_level = 0.0
    reason = "No strong consensus reached."

    # PRIORITY 1: Custom Model (M1) - The specialized Indian context expert
    # Threshold is set at 75.0% to prioritize the specific, highly relevant model.
    if m1_res['prediction'] == "FAKE" and m1_res['confidence'] >= 75.0: 
        final_verdict = "FAKE"
        confidence_level = m1_res['confidence']
        reason = f"Custom TF-IDF Model found a high-confidence pattern ({m1_res['confidence']}%) trained on BharatKosh data."
    
    # PRIORITY 2: Web Search (M2) - External fact-checking verification
    elif m2_res['flagged']:
        final_verdict = "FAKE"
        confidence_level = 90.0 # High fixed confidence due to external verification
        reason = "Fact-checking websites found similar content debunked."
    
    # PRIORITY 3: General AI (M3) - Linguistic style check is the final tie-breaker
    else:
        final_verdict = m3_res['prediction']
        confidence_level = round(m3_res['confidence'] * 100, 1)
        reason = f"General BERT Style Analysis indicates {final_verdict} ({confidence_level}%)."


    return {
        "verdict": final_verdict,
        "confidence": confidence_level,
        "reason": reason,
        "breakdown": {
            "custom_model": m1_res, 
            "web_search": m2_res,
            "ai_model": m3_res
        }
    }