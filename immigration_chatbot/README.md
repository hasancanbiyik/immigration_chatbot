# US Immigration Q&A Chatbot

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, end-to-end chatbot designed to answer common questions about the US immigration process, including H-1B, OPT, Green Cards, and F-1/CPT visas. This project leverages a powerful sentence-embedding model for accurate semantic search and is deployed with a user-friendly web interface built in Streamlit.

---

## 🎥 Live Demo

> **Note:** A live demo or a GIF is one of the most effective ways to showcase your project. You can use a tool like Giphy Capture or ScreenToGif to record your application and embed it here.

## ✨ Key Features

* **Semantic Search:** Moves beyond simple keyword matching by using a `SentenceTransformer` model to understand the *semantic meaning* of a user's question, providing highly relevant answers.
* **Dynamic Knowledge Base:** The chatbot's knowledge is stored in a structured `qa_data.json` file, allowing for easy updates and expansion without changing the application code.
* **Interactive UI:** A clean and responsive user interface built with Streamlit, providing a seamless chat experience.
* **Contextual Follow-Ups:** Suggests relevant follow-up questions to guide the user's conversation flow.
* **User Feedback Mechanism:** Allows users to rate answers as "Helpful" or "Not Helpful," providing a mechanism for future model improvement and evaluation.
* **Admin & Session Management:** Includes features for clearing the chat history, viewing session statistics, and downloading the conversation log.

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
