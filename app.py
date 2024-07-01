import streamlit as st
from langchain_backend import ContentEngine

st.title("Content Engine for Comparing PDF Documents")

content_engine = ContentEngine()

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.write("Processing uploaded files...")
    for uploaded_file in uploaded_files:
        text = content_engine.parse_pdf(uploaded_file)
        chunks = content_engine.split_text(text)
        embeddings = content_engine.generate_embeddings(chunks)
        content_engine.store_vectors(embeddings, chunks)
    st.write("Files processed successfully.")

query = st.text_input("Enter your query")
if query:
    st.write("Searching for relevant documents...")
    results = content_engine.retrieve_documents(query)
    if results:
        insights = content_engine.generate_insights(query, results)
        st.write(insights)
    else:
        st.write("No relevant documents found.")
