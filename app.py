import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (first time only)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

print(stop_words)
# -------------------------------
# 🔧 Text Preprocessing Function
# -------------------------------
def preprocess(text):
    text = text.lower()  # lowercase

    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove special chars

    words = text.split()

    # remove stopwords + lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    print(words)
    return " ".join(words)

# -------------------------------
# 📊 Load Data
# -------------------------------
df = pd.read_csv("abap_qa.csv")

# Preprocess questions
df["clean_question"] = df["question"].apply(preprocess)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_question"])

# -------------------------------
# 🤖 Get Answer Function
# -------------------------------
def get_answer(user_query):
    clean_query = preprocess(user_query)

    user_vec = vectorizer.transform([clean_query])

    similarity = cosine_similarity(user_vec, X)

    index = similarity.argmax()
    score = similarity[0][index]

    return df.iloc[index]["answer"], score

# -------------------------------
# 🎨 Streamlit UI
# -------------------------------
st.title("🤖 ABAP NLP Assistant")

user_input = st.text_input("💬 Ask your ABAP question:")

if st.button("Get Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a question")
    else:
        answer, score = get_answer(user_input)

        st.subheader("✅ Answer:")
        st.write(answer)

        st.write(f"📊 Confidence Score: {round(score*100,2)}%")