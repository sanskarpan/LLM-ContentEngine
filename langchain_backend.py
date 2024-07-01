import PyPDF2
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import faiss

class ContentEngine:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Sentence embeddings model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # GPT-2 tokenizer
        self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')  # GPT-2 model for text generation
        self.dimension = 384  # Embedding size for Sentence Transformer
        self.index = faiss.IndexFlatL2(self.dimension)  # Faiss index for document vectors

    def parse_pdf(self, pdf_file):
        # Function to parse text from a PDF file
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def split_text(self, text, chunk_size=500, overlap=100):
        # Function to split text into chunks for processing
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if len(chunk) < chunk_size and chunks:
                chunks[-1] += chunk
                break
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks
    
    def generate_embeddings(self, texts):
        # Function to generate embeddings for chunks of text
        return self.model.encode(texts)
    
    def store_documents(self, documents, embeddings):
        # Function to store documents and their embeddings in memory
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        # Build Faiss index for fast retrieval
        self.index.add(np.array(embeddings))
    
    def retrieve_documents(self, query, top_k=3):
        # Function to retrieve relevant documents based on query
        query_embedding = self.model.encode([query])[0]
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        results = [(self.documents[idx], distances[0][i]) for i, idx in enumerate(indices[0]) if idx < len(self.documents)]
        return results
    
    def generate_insights(self, query, results):
        # Function to generate insights based on query and retrieved documents
        prompt = f"Question: {query}\n\n"
        for doc, dist in results:
            prompt += f"Document: {doc[:500]}\n\n"

        # Use GPT-2 for generating response
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        outputs = self.gpt_model.generate(
            inputs['input_ids'],
            max_length=512,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Post-processing the response to remove redundant lines
        response = self.post_process_response(response)
        return response
    
    def post_process_response(self, response):
        # Function to remove redundant lines from the generated response
        lines = response.split('\n')
        seen = set()
        filtered_lines = []
        for line in lines:
            if line.strip() not in seen:
                seen.add(line.strip())
                filtered_lines.append(line)
        return '\n'.join(filtered_lines)

