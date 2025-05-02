# 🤖 SmartHelp: Generative AI for Scalable Customer Service

**SmartHelp** is an intelligent, scalable AI-driven assistant that automates context-aware email responses. Powered by Retrieval-Augmented Generation (RAG), semantic search, and multilingual support, it boosts customer service efficiency by delivering personalized, human-like responses with minimal manual intervention.

---

## ✨ Features

| Feature                       | Functionality                                                                 |
|------------------------------|--------------------------------------------------------------------------------|
| **Intent Detection & Sentiment Analysis** | Understands user intent and urgency using GenAI – flags negative tone for escalation. |
| **Relevant Context Fetching**           | Pulls accurate, query-specific information from stored knowledge to support replies. |
| **Custom Response Generation**         | Uses GPT-3o Mini to generate personalized replies instead of fixed templates. |
| **Semantic Search Engine**             | Matches user queries with stored content to retrieve the most relevant answers. |
| **Multilingual Support**               | Can process and respond to queries in multiple languages. |
| **24/7 Availability**                  | Always-on assistant that scales with demand, even during high-volume periods. |

---

## 🛠️ Tools & Technologies Used

| Tool / Technology           | Purpose & Why We Used It                                                                 |
|----------------------------|-------------------------------------------------------------------------------------------|
| **RAG (Retrieval-Augmented Generation)** | Enhances response accuracy by grounding generative AI replies in relevant knowledge base content. |
| **ChromaDB**                | A lightweight, high-performance vector database used for fast and efficient semantic search and retrieval. |
| **BERT (base-uncased)**     | Pre-trained language model used for extracting intent and contextual meaning from user queries. |
| **MarianMT Translator**     | Provides multilingual support by translating user inputs and generated responses using neural machine translation. |
| **Streamlit**               | Enables rapid development of a user-friendly web interface for internal testing and live demonstrations. |

---

## 🚀 Sample Workflow

1. **User Email Received** – A customer sends an inquiry via email.
2. **Translation (if needed)** – Input is translated into the english if the original input is not in english.
3. **Intent & Tone Detected** – System analyzes intent (e.g., complaint, request) and sentiment (e.g., frustration).
4. **Context Retrieved** – Relevant support documents are fetched using semantic search (ChromaDB).
5. **Response Generated** – GPT-based engine generates a personalized response grounded in retrieved context.
6. **Email Sent + Logged** – The response is sent, and the conversation is logged for future learning.

---

## 📈 Future Enhancements

- Add human-in-the-loop verification for high-stakes responses.
- Extend knowledge base auto-updates via web scrapers or CMS sync.
- Voice-to-text query support for phone-based customer service.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributors

Made with ❤️ by Arjun, Anubhav, Shubha, Saurabh & Sujoy (MSIS Class of Fall 2024)

---