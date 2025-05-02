import streamlit as st
import torch
import pandas as pd
#from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline, BertForSequenceClassification, BertTokenizer
from transformers import MarianMTModel, MarianTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from langdetect import detect
from chromadb.config import Settings
import chromadb
import warnings
warnings.filterwarnings("ignore")
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPEN_AI_API_KEY")
client = OpenAI(api_key=api_key)

# ========== Setup ==========

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
classification_model = BertForSequenceClassification.from_pretrained("my_model").to(device)
classification_tokenizer = BertTokenizer.from_pretrained("my_tokenizer")

translator_model_name = 'Helsinki-NLP/opus-mt-mul-en'
translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
translator_model = MarianMTModel.from_pretrained(translator_model_name).to(device)

sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_name).to(device)
sentiment_tokenizer = BertTokenizer.from_pretrained(sentiment_model_name)

# Label Maps
label_map = {
    "Billing and Payments": 0,
    "Customer Service": 1,
    "General Inquiry": 2,
    "Human Resources": 3,
    "IT Support": 4,
    "Product Support": 5,
    "Returns and Exchanges": 6,
    "Sales and Pre-Sales": 7,
    "Service Outages and Maintenance": 8,
    "Technical Support": 9
}
reverse_label_map = {v: k for k, v in label_map.items()}

# Load dataset and build vector store
df = pd.read_csv("SupportDataset.csv")
df.dropna(subset=["subject", "body", "answer"], inplace=True)
df["text"] = df["subject"] + " " + df["body"]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_client = chromadb.Client(Settings())

try:
    vectorstore = Chroma(persist_directory="db", embedding_function=embedding_model, client=chroma_client)
    print("ChromaDB loaded successfully.")
except:
    vectorstore = Chroma.from_texts(texts=df["text"].tolist(), embedding=embedding_model, persist_directory="db")
    vectorstore.persist()
    print("ChromaDB created and persisted successfully.")

retriever = vectorstore.as_retriever()

# Create GPT-2 Pipeline for Response
generator_pipeline = pipeline(
    "text-generation",
    model="gpt2",  # or whatever your model is
    device=0 if torch.cuda.is_available() else -1,
    max_new_tokens=256,   # <-- Important
    do_sample=True,
    temperature=0.7,
)

llm = HuggingFacePipeline(pipeline=generator_pipeline)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# ========== Helper Functions ==========

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text):
    inputs = translator_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    translated = translator_model.generate(**inputs)
    translated_text = translator_tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

def classify_email(subject, body):
    text = subject + " " + body
    inputs = classification_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = classification_model(**inputs)
    predicted_class = torch.argmax(outputs.logits, axis=-1).item()
    return reverse_label_map[predicted_class]

def detect_sentiment(text):
    sentiment_labels = {
        0: "Very Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Very Positive"
    }
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = sentiment_model(**inputs)
    sentiment_score = torch.argmax(outputs.logits, axis=-1).item()
    sentiment_reason = sentiment_labels.get(sentiment_score, "Unknown")
    return sentiment_score, sentiment_reason

def find_best_past_answer(query):
    docs = retriever.get_relevant_documents(query)
    if docs:
        best_doc = docs[0]
        print("best_doc:", best_doc)
        return best_doc.page_content  # assumes 'answer' or combined text is in page_content
    if not docs or docs[0].metadata.get("relevance", 0) < 0.5:
        return "This inquiry does not appear related to supported topics. Please contact support for assistance."
    
def generate_response_with_openai(subject, body, best_past_answer):
    prompt = f"""You are a helpful customer support assistant.

    A customer emailed the following:

    Subject: {subject}
    Body: {body}

    Here is a past answer that solved a similar issue:
    "{best_past_answer}"

    Using the context above, draft a clear and helpful reply to the customer. Keep it professional and easy to understand.

    Reply:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional customer support assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=256
    )

    return response.choices[0].message.content.strip()

# ========== Streamlit App ==========

st.title("📧 Automated Email Response System")

st.write("This app detects email language, translates if needed, classifies intent, detects sentiment, and generates a response!")

# Input fields
customer_email = st.text_input("Customer Email")
email_subject = st.text_input("Email Subject")
email_body = st.text_area("Email Body")

if st.button("Process Email"):
    if not email_subject or not email_body:
        st.warning("Please provide both Subject and Body.")
    else:
        full_text = email_subject + " " + email_body

        # 1. Language Detection
        detected_language = detect_language(full_text)

        # 2. Translation if needed
        if detected_language != "en":
            email_subject = translate_to_english(email_subject)
            email_body = translate_to_english(email_body)

        # 3. Email Classification
        category = classify_email(email_subject, email_body)

        # 4. Sentiment Detection
        sentiment_score, sentiment_reason = detect_sentiment(email_body)

        # 5. Retrieve best past answer and generate GPT response
        query_text = f"{email_subject} {email_body}"
        best_past_answer = find_best_past_answer(query_text)
        result_text = generate_response_with_openai(email_subject, email_body, best_past_answer)

        # Step 3: Build the final cleaned output
        final_response = {
            "query": query_text,
            "result": result_text
        }

        # Step 6: Save conversation into ChromaDB
        try:
            combined_text = f"Subject: {email_subject}\nBody: {email_body}\nResponse: {result_text}"
            vectorstore.add_texts([combined_text])
            vectorstore.persist()
        except Exception as e:
            st.error(f"Error saving conversation: {str(e)}")

        # Output
        st.success("✅ Email Processed Successfully!")
        st.write(f"**Detected Language:** {detected_language}")
        st.write(f"**Predicted Category:** {category}")
        st.write(f"**Detected Sentiment:** {sentiment_reason} (Score: {sentiment_score})")
        st.write(f"**Auto Response:** {final_response.get('result')}")
        
        stored_docs = vectorstore.get()["documents"]
        for i, doc in enumerate(stored_docs):
            print(f"[{i+1}] {doc}")