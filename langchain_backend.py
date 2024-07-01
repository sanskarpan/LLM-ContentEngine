import PyPDF2
from sentence_transformers import SentenceTransformer
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import faiss
import numpy as np

class ContentEngine:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.dimension = 384  # Embedding size for 'all-MiniLM-L6-v2'
        self.index = faiss.IndexFlatL2(self.dimension)

    def parse_pdf(self, pdf_file):
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def split_text(self, text, chunk_size=500, overlap=100):
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
        return self.model.encode(texts)
    
    def store_vectors(self, vectors, texts):
        self.index.add(np.array(vectors))
        self.documents.extend(texts)
    
    def retrieve_documents(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)
        results = [(self.documents[idx], distances[0][i]) for i, idx in enumerate(indices[0]) if idx < len(self.documents)]
        return results
    
    def generate_insights(self, query, results):
        prompt = f"Question: {query}\n\n"
        for doc, dist in results:
            prompt += f"Document: {doc[:500]}\n\n"

        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        outputs = self.gpt_model.generate(
            inputs['input_ids'],
            max_new_tokens=150,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Post-processing the response to remove repetitive text
        response = self.post_process_response(response)
        return response

    def post_process_response(self, response):
        # Removing redundant or repetitive lines
        lines = response.split('\n')
        seen = set()
        filtered_lines = []
        for line in lines:
            if line.strip() not in seen:
                seen.add(line.strip())
                filtered_lines.append(line)
        return '\n'.join(filtered_lines)
