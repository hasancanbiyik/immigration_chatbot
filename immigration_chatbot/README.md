# US Immigration Q&A Chatbot

## 🏗️ System Architecture

This application is composed of several key components that work together to deliver an intelligent Q&A experience.

```mermaid
graph TD
    A[User] --> B{Streamlit Frontend};
    B --> C[Chatbot Logic Engine];
    C -- Encodes User Query --> D(SentenceTransformer Model);
    C -- Finds Most Similar Question --> E[Vector Embeddings Corpus];
    E -- Built from --> F(qa_data.json);
    C -- Retrieves Answer --> F;
    C --> B;
