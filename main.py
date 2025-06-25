import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

# Cargar el archivo de preguntas
df = pd.read_csv("fqs.csv")
questions = df["question"].tolist()
answers = df["answer"].tolist()

# Cargar el modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generar embeddings para las preguntas del CSV
embeddings = model.encode(questions)

# Guardar los datos para futuros usos (opcional)
with open("faq_data.pkl", "wb") as f:
    pickle.dump({"questions": questions, "answers": answers, "embeddings": embeddings}, f)

# Configurar el buscador con scikit-learn (cosine similarity)
searcher = NearestNeighbors(n_neighbors=1, metric="cosine")
searcher.fit(embeddings)

# Función para responder preguntas
def responder(pregunta_usuario):
    embedding_usuario = model.encode([pregunta_usuario])
    distancia, indice = searcher.kneighbors(embedding_usuario)
    idx = indice[0][0]
    similitud = 1 - distancia[0][0]

    print(f"\n Pregunta del usuario: {pregunta_usuario}")
    print(f"Respuesta encontrada: {answers[idx]}")
    print(f"Similitud coseno: {similitud:.4f}")

# Ejemplo interactivo
if __name__ == "__main__":
    while True:
        pregunta = input("\nEscribe tu pregunta (o 'exit'): ")
        if pregunta.lower() == "exit":
            break
        responder(pregunta)
