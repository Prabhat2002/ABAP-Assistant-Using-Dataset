# -------------------------------
# 📦 Import Libraries
# -------------------------------
import nltk
import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 📥 NLTK Setup
# -------------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------------
# 🔹 Preprocessing
# -------------------------------
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]

    return tokens

# -------------------------------
# 🔹 Lemmatization
# -------------------------------
def lemmatization_text(tokens):
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# -------------------------------
# 📊 Load Data
# -------------------------------
df = pd.read_csv("abap_qa.csv")

# Clean dataset
df["clean_question"] = df["question"].apply(
    lambda x: lemmatization_text(preprocess_text(x))
)

# -------------------------------
# 📐 TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_question"])

# -------------------------------
# 💾 Session State (Chat Memory)
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# -------------------------------
# 🤖 Get Answer (with context)
# -------------------------------
def get_answer(user_query):

    # Follow-up handling
    if st.session_state.last_query:
        combined_query = st.session_state.last_query + " " + user_query
    else:
        combined_query = user_query

    # NLP pipeline
    tokens = preprocess_text(combined_query)
    clean_query = lemmatization_text(tokens)

    user_vec = vectorizer.transform([clean_query])
    similarity = cosine_similarity(user_vec, X)

    index = similarity.argmax()
    score = similarity[0][index]

    answer = df.iloc[index]["answer"]

    # Save context
    st.session_state.last_query = user_query

    return answer, score

# -------------------------------
# 🎨 UI
# -------------------------------
st.title("🤖 ABAP Chat Assistant")

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

# Chat input (like ChatGPT)
user_input = st.chat_input("Ask your ABAP question...")

if user_input:
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # Get bot answer
    answer, score = get_answer(user_input)

    response = f"{answer}\n\n📊 Confidence: {round(score*100,2)}%"

    # Show bot message
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.write(response)

# -------------------------------
# 🔄 Reset Button
# -------------------------------
if st.button("Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.last_query = ""