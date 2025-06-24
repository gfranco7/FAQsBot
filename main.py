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

"""
print("Bienvenido a Amazun\n")
user_input = input("Input your question: ")

embedding = model.encode([user_input])

D, I= index.search(embedding, k=1)
"""

eval_df = pd.read_csv("eval_set.csv")

correct = 0

for i, row in eval_df.iterrows():
    user_q = row["user_question"]
    expected_a = row["expected_answer"]
    
    embedding = model.encode([user_q])
    D, I= index.search(embedding, k=3)
    idx = I[0][0]
    predicted_a = answers[idx]

    print(f"\nPregunta: {user_q}")
    print(f"Respuesta obtenida: {predicted_a}")
    print(f"ESperada: {expected_a}")

    if expected_a.strip().lower() in predicted_a.strip().lower():
        correct+=1

accuracy = correct / len(eval_df)
print(f"Precision: {accuracy:.2f}")


"""
if distancia < 1.0:
    print("||")
    print(f"Te efieres a: {questions[idx]}")
    print(f"Respuesta: {answers[idx]}")
else:
    print("||")
    print("No se encontró una respuesta adecuada")


"""