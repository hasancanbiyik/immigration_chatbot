# U.S. Immigration Q&A Chatbot 🇺🇸🤖🦅
## Project Overview
An intelligent, end-to-end chatbot designed to answer common questions about the US immigration process, including H-1B, OPT, Green Cards, and F-1/CPT visas. This project leverages a powerful sentence-embedding model for accurate semantic search and is deployed with a user-friendly web interface built in Streamlit.

### Live Demo
A visual demonstration will be added here [Noted on July 9, 2025]

## Key Features
- **Semantic Search:** Moves beyond simple keyword matching by using a SentenceTransformer model (multi-qa-mpnet-base-dot-v1) to understand the semantic meaning of a user's question, ensuring highly relevant answers.
- **Dynamic Knowledge Base:** The chatbot's intelligence is stored in a structured qa_data.json file. This allows for easy updates, expansion, and fine-tuning of the knowledge base without altering the application's source code.
- **Interactive UI:** A clean and responsive user interface built with Streamlit provides a seamless chat experience, complete with user feedback buttons (Helpful/Not Helpful) to capture response quality.
- **Contextual Conversation Flow:** The chatbot suggests relevant follow-up questions after each answer, guiding the user to related topics and creating a more natural and helpful interaction.
- **Session Management & Admin Tools:** Includes features for clearing the chat history, viewing real-time session statistics, and downloading the full conversation log as a JSON file.

## Files
**1. Streamlit Frontend (`app.py`):** Captures user input and displays the conversation history. It manages the session state and user interactions like button clicks and feedback.

**2. Chatbot Logic Engine (`chatbot/logic.py`):** The core of the application. It orchestrates the process of receiving a query, generating an embedding, performing the search, and retrieving the appropriate response.

**3. SentenceTransformer Model (`multi-qa-mpnet-base-dot-v1`):** A pre-trained NLP model that converts both the user queries and the knowledge base questions into high-dimensional vector embeddings.

**4. Knowledge Base (`chatbot/qa_data.json`):** A JSON file containing topics, representative questions for semantic matching, answers, and follow-up suggestions.


## Getting Started
Follow these instructions to set up and run the project locally:

#### Prerequisites
- Python 3.9 or higher
- `pip` and `venv`

#### Installation & Setup

**1. Clone the repository:**
`git clone [https://github.com/hasancanbiyik/immigration_chatbot]`

`cd hasancanbiyik`

**2. Create and activate a virtual environment:**

##### For macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

##### For Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```

**3. Install the required dependencies:**

A `requirements.txt` file is included for easy installation.

`pip install -r requirements.txt`

**4. Lastly, run the Streamlit application:**

`streamlit run app.py`

The application should now be open and running in your web browser!

```text
📁 Project Structure (immigration_chatbot)
├── chatbot/
│   ├── __init__.py         # Core chatbot class and semantic search logic
│   ├── logic.py            # Core chatbot class and semantic search logic
│   └── qa_data.json        # Knowledge base with questions and answers
├── app.py                  # Main Streamlit application file
├── README.md               # You are here!
└── requirements.txt        # Project dependencies
```

**Bonus:** Check out system_architecture.md for a cool visualization of how this project works and let me know what you think!

## Future Improvements
This project serves as a strong foundation. Future enhancements could include:

- [ ] **Evaluation Framework:** Implement evaluation methods to test chatbot accuracy on held-out sets.
- [ ] **Vector DB:** Swap in FAISS or ChromaDB for scalable vector search.
- [ ] **Model Fine-Tuning:** Train on a domain-specific dataset to boost precision.
- [ ] **RAG Upgrade:** Move to Retrieval-Augmented Generation using Hugging Face models.
- [ ] **Dockerization:** Add a `Dockerfile` for deployment consistency.

## License
This project is licensed under the MIT License.





