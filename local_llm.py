from transformers import pipeline

class LocalLLM:
    def __init__(self):
        self.pipeline = pipeline('text-generation', model='distilgpt2')

    def generate_insights(self, query):
        return self.pipeline(query, max_length=100)
