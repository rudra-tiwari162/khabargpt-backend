import pandas as pd
import re
import nltk
import joblib  # To save the trained model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- 1. SETUP & DOWNLOADS ---
print("⏳ Downloading NLTK data...")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- 2. LOAD DATA ---
# Make sure the file name matches exactly what you have in your 'data' folder
file_path = 'data/BharatFakeNewsKosh.xlsx' 
print(f"⏳ Loading dataset from {file_path}...")

try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"❌ Error: Could not find {file_path}. Please check the file name and location.")
    exit()

# --- 3. PREPROCESSING (Your Code) ---
print("🧹 Cleaning text data...")

# Keep only needed columns
# NOTE: Ensure your Excel actually has these column names. 
# If your Excel has 'Title' or 'Content', change 'Eng_Trans_News_Body' below.
df = df[['Eng_Trans_News_Body', 'Label']]

# Drop missing rows
df = df.dropna(subset=['Eng_Trans_News_Body', 'Label'])

# Map labels: Ensure we handle string or boolean types
df['Label'] = df['Label'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})

# Drop rows where Label parsing failed (if any)
df = df.dropna(subset=['Label'])

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply cleaning
df['clean_text'] = df['Eng_Trans_News_Body'].apply(clean_text)

print(f"✅ Data processed. Total samples: {len(df)}")

# --- 4. TF-IDF VECTORIZATION ---
print("🧮 Converting text to numbers (TF-IDF)...")

# max_features=5000 keeps the model lightweight and fast
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2)) 
X = vectorizer.fit_transform(df['clean_text'])
y = df['Label']

# --- 5. MODEL TRAINING ---
print("🧠 Training Logistic Regression Model...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression is excellent for text classification
model = LogisticRegression()
model.fit(X_train, y_train)

# --- 6. EVALUATION ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {acc*100:.2f}%")

# --- 7. SAVE THE BRAIN ---
print("💾 Saving model and vectorizer...")
joblib.dump(model, 'method1_model.pkl')
joblib.dump(vectorizer, 'method1_vectorizer.pkl')
joblib.dump(clean_text, 'method1_cleaner.pkl') # Saving the function can be tricky, we'll redefine it in main.py instead usually

print("✅ DONE! 'method1_model.pkl' and 'method1_vectorizer.pkl' created.")