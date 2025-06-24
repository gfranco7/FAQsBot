import pandas as pd 
import faiss
import pickle
from sentence_transformers import SentenceTransformer, util

csv_file = pd.read_csv('fqs.csv')
# print(csv_file.head())

model = SentenceTransformer("All-MiniLM-L6-v2")
# index = faiss.read_index("faq_index.faiss")

