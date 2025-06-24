import os
import pandas as pd 
import faiss
import pickle
from sentence_transformers import SentenceTransformer, util


CSV_PATH = "fqs.csv"
INDEX_PATH = "faq_index.faiss"
DATA_PATH = "faq_data.pkl"

if not os.path.exists(CSV_PATH):
    print(f"No se encontró el archivo {CSV_PATH}")


df = pd.read_csv(CSV_PATH)
questions = df["question"].tolist()
answers = df["answer"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)

print("Bienvenido a Amazun\n")
user_input = input("Input your question: ")

embedding = model.encode([user_input])

D, I= index.search(embedding, k=1)

idx = I[0][0]
distancia = D[0][0]

if distancia < 1.0:
    print("||")
    print(f"Te efieres a: {questions[idx]}")
    print(f"Respuesta: {answers[idx]}")
else:
    print("||")
    print("No se encontró una respuesta adecuada")

