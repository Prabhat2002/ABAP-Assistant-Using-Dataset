import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st  
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer 

# -------------------------------
# 📥 NLTK Download
# -------------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------------
# 🔹 METHOD 1: Preprocessing
# -------------------------------
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens   # list return

# -------------------------------
# 🔹 METHOD 2: Lemmatization
# -------------------------------
def lemmatization_text(tokens):   # ✅ FIX: text → tokens
    lemmatizer = WordNetLemmatizer()
    # ✅ FIX: loop over tokens (pehle pura list pass ho raha tha)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)   # ✅ FIX: string return (important)

# -------------------------------
# 📊 Load Data
# -------------------------------
df = pd.read_csv("abap_qa.csv")

# ✅ FIX: preprocessing + lemmatization dono apply karo
df["question"] = df["question"].apply(
    lambda x: lemmatization_text(preprocess_text(x))
)

# -------------------------------
# 📐 TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])   # ab string mil raha hai ✅

# -------------------------------
# 🤖 Answer Function
# -------------------------------
def get_answer(user_query):
    
    # ✅ FIX: preprocess → tokens
    tokens = preprocess_text(user_query)

    # ✅ FIX: tokens → string after lemmatization
    clean_query = lemmatization_text(tokens)

    # ✅ FIX: same vectorizer use karo (naya TF-IDF mat banao)
    user_vec = vectorizer.transform([clean_query])

    similarity = cosine_similarity(user_vec, X)

    index = similarity.argmax()
    score = similarity[0][index]
    return df.iloc[index]["answer"], score

# -------------------------------
# 🎨 Streamlit UI
# -------------------------------
st.title("🤖 ABAP NLP Assistant")

text_data = st.text_input("💬 Ask your ABAP question:")

if st.button("Get Answer"):
    if text_data.strip() == "":
        st.warning("Please enter a question")
    else:
        answer, score = get_answer(text_data)
        st.subheader("✅ Answer:")
        if score < 0.8:  # ✅ FIX: confidence threshold
            st.write(answer)
            st.warning(f"📊 Confidence Score: {round(score*100,2)}%"  )
        else:
            st.write(answer)
            st.success(f"📊 Confidence Score: {round(score*100,2)}%")