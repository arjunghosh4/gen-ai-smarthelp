import imapclient
import email
import smtplib
import torch
from email.message import EmailMessage
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

# 🔹 Configure Email Settings
IMAP_SERVER = os.getenv("IMAP_SERVER") 
IMAP_PORT = os.getenv("IMAP_PORT") 
EMAIL_USER = os.getenv("EMAIL_USER") 
EMAIL_PASS = os.getenv("EMAIL_PASS") 

SMTP_SERVER = os.getenv("SMTP_SERVER") 
SMTP_PORT = os.getenv("SMTP_PORT") 

# 🔹 Load AI Model & Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("./my_model").to(device)
tokenizer = BertTokenizer.from_pretrained("./my_tokenizer")

df = pd.read_csv("SupportDataset.csv").dropna(subset=["subject", "body", "answer"])
df["text"] = df["subject"] + " " + df["body"]
qa_pairs = list(zip(df["text"], df["answer"]))

vectorizer = TfidfVectorizer(stop_words="english")
question_vectors = vectorizer.fit_transform([q for q, _ in qa_pairs])

def find_best_past_answer(user_question):
    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarities.argmax()
    return qa_pairs[best_match_index][1]

def process_email(subject, body):
    text = subject + " " + body
    best_answer = find_best_past_answer(text)
    return best_answer if best_answer else "Thank you for your email. Our team will get back to you soon."

def send_email(to_email, subject, body):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = to_email
    msg.set_content(body) #message details

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
            print(f"✅ Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Failed to send email: {str(e)}")
# List of keywords to filter relevant emails
KEYWORDS = ["issue", "billing", "technical", "integration", "data breach"]

from datetime import datetime, timedelta

# Define how many days back to check
DAYS_BACK = 1  # Change to 1 day, so only today’s emails are checked

def check_for_new_emails():
    try:
        with imapclient.IMAPClient(IMAP_SERVER, port=IMAP_PORT) as client:
            client.login(EMAIL_USER, EMAIL_PASS)
            client.select_folder("INBOX", readonly=False)

            # Get today's date minus DAYS_BACK
            date_since = (datetime.now() - timedelta(days=DAYS_BACK)).strftime("%d-%b-%Y")
            
            # Search for unread emails from the last X days
            messages = client.search([u'UNSEEN', u'SINCE', date_since])  
            
            for msg_id in messages:
                raw_message = client.fetch(msg_id, ["RFC822"])[msg_id][b"RFC822"]
                email_msg = email.message_from_bytes(raw_message)
                
                subject = email_msg["subject"].lower() if email_msg["subject"] else ""
                sender = email_msg["from"]
                body = ""

                if email_msg.is_multipart():
                    for part in email_msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode()
                            break
                else:
                    body = email_msg.get_payload(decode=True).decode()

                print(f"📧 New Email from {sender}: {subject}")

                # Step 1: Check if email is relevant
                if any(keyword in subject for keyword in KEYWORDS):
                    print(f"✅ Relevant email detected! Processing...")
                    
                    # Step 2: Process email using AI
                    response_text = process_email(subject, body)

                    # Step 3: Send response
                    send_email(sender, f"Re: {subject}", response_text)

                    # Step 4: Mark email as read
                    client.add_flags(msg_id, "\\Seen")
                else:
                    print(f"❌ Ignoring email - No relevant keywords found.")
                    
    except Exception as e:
        print(f"❌ Error checking emails: {str(e)}")

# 🔹 Run the Email Checker Every Few Minutes
if __name__ == "__main__":
    import time
    while True:
        print("🔄 Checking for new emails...")
        check_for_new_emails()
        time.sleep(20)  # Check inbox every 60 seconds
