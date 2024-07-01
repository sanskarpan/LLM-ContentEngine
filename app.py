import streamlit as st
from langchain_backend import ContentEngine

# Initialize ContentEngine instance
content_engine = ContentEngine()
st.title("Content Engine for PDF Documents")
st.sidebar.title("Options")

# File uploader in the sidebar to upload PDF files
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Processing uploaded files if any are uploaded
if uploaded_files:
    st.sidebar.write("Processing uploaded files...")

    # Processing files asynchronously with a spinner
    with st.spinner('Processing...'):
        for uploaded_file in uploaded_files:
            # Extract text from each uploaded PDF file,then splitting text into manageable chunks for analysis
            text = content_engine.parse_pdf(uploaded_file)
            chunks = content_engine.split_text(text)
            # Generate embeddings for each chunk of text & store documents and their embeddings in ContentEngine
            embeddings = content_engine.generate_embeddings(chunks)
            content_engine.store_documents(chunks, embeddings)

    # Display success message in the sidebar after processing
    st.sidebar.success("Files processed successfully!")

# Text input field to enter a query for document retrieval
query = st.text_input("Enter your query")
if query:
    st.write("Searching for relevant data...")

    # Retrieve documents based on the query
    results = content_engine.retrieve_documents(query)
    if results:
        # Generate insights or summaries based on retrieved documents
        insights = content_engine.generate_insights(query, results)
        st.write(insights[:1000])
    else:
        # Display message if no relevant documents are found
        st.write("No relevant documents found.")
