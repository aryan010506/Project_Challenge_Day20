# -------------------------------------------------
# Day 20 Project - Email Classifier with File Upload
# By Aryan Sunil
# -------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# -----------------------------
# STREAMLIT PAGE SETTINGS
# -----------------------------
st.set_page_config(page_title="Email Classifier", page_icon="📧")
st.title("📧 Email Classifier (Day 20 Project)")

# -----------------------------
# UPLOAD CSV FILE
# -----------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load the uploaded CSV
    data = pd.read_csv(uploaded_file, encoding='latin-1', on_bad_lines='skip')
    # Many Kaggle spam CSVs have extra columns — keep only two
    if 'v1' in data.columns and 'v2' in data.columns:
        data = data[['v1', 'v2']]
        data.columns = ['label', 'message']
    else:
        st.error("CSV must contain columns 'v1' (label) and 'v2' (message)")
        st.stop()

    # Convert labels to 0/1
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
else:
    st.info("Please upload your CSV file to start.")
    st.stop()

# -----------------------------
# SPLIT DATA
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# -----------------------------
# TF-IDF VECTORIZATION
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# CLASSIFICATION FUNCTION
# -----------------------------
def classify_email(text: str) -> str:
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]
    return "🚨 Spam" if pred == 1 else "✅ Not Spam"

# -----------------------------
# STREAMLIT UI FOR CLASSIFICATION
# -----------------------------
st.write(f"**Model Accuracy:** {accuracy:.2%}")

email_text = st.text_area("Paste your email content here:")

if st.button("Classify"):
    if email_text.strip() == "":
        st.warning("Please enter an email to classify.")
    else:
        result = classify_email(email_text)
        if "Spam" in result:
            st.error(f"Result: {result}")
        else:
            st.success(f"Result: {result}")

st.caption("Built with scikit-learn + Streamlit | Day 20 Challenge")

# -----------------------------
# (Optional) Show classification report
# -----------------------------
with st.expander("See classification report"):
    st.text(classification_report(y_test, y_pred))
