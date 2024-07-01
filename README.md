# LLM Content Engine for PDF Documents

This application utilizes LangChain and Streamlit to provide a streamlined interface for processing and querying PDF documents. It integrates document parsing, embedding generation, and query-based document retrieval to facilitate efficient data exploration and insights extraction.

## Features

- **Upload PDF Files:** Users can upload multiple PDF files for processing.
- **Text Chunking:** Automatically splits PDF text into manageable chunks for analysis.
- **Document Embeddings:** Generates embeddings using Sentence Transformers for semantic representation.
- **Query-Based Retrieval:** Retrieves relevant documents based on user-entered queries.
- **Insights Generation:** Utilizes GPT-2 for generating insights or summaries from retrieved documents.
## Dependencies
- **PyPDF2:** Used for parsing PDF documents.
- **Sentence Transformers:** Provides embeddings for text.
- **Transformers (GPT2Tokenizer, GPT2LMHeadModel):** Tokenization and generation of text using GPT-2.
- **Faiss:** Vector index for efficient document retrieval.
- **Streamlit:** Frontend interface for user interaction.
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sanskarpan/LLM-ContentEngine.git
    ```
2. Install dependencies:
    ```bash
   pip install -r requirements.txt
    ```
## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
     ```
2. Access the app in your browser at `http://localhost:8501`.

