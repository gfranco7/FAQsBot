import pandas as pd     
from sentence_transformers import SentenceTransformer, util  #Modelo transformer
from transformers import pipeline  #Para búsqueda semántica
from sklearn.cluster import KMeans #Para agrupar los embedders
import numpy as np




model = SentenceTransformer("All-MiniLM-L6-v2")


emb1 = model.encode("Cristtiano Ronaldo es el mejor")
emb2 = model.encode("El real madrid es el equipo mas grande")

cos_sim= util.cos_sim(emb1, emb2)  # ==== Se obtiene el la similitud coseno

# print("THis is the Cosine-Similarity: ", cos_sim) 

sentences = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.'
          ]


embeddings = model.encode(sentences) # === codifica las cadenas
cos_sim = util.cos_sim(embeddings, embeddings)

all_sentence_combinations = []

for i in range(len(cos_sim)):
    for j in range(i+1, len(cos_sim)):
        all_sentence_combinations.append((cos_sim[i][j],i,j))

# print(all_sentences_combinatios)


#Sort list by the highest cosine similarity score
all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

# print("Top-5 most similar pairs:")
# for score, i, j in all_sentence_combinations[0:5]:
#     print("{} \t {} \t {:.4f}".format(sentences[i], sentences[j], cos_sim[i][j]))


# ==============================================================================

model_faq =SentenceTransformer('clips/mfaq')

question = "How many models can I host on HuggingFace?"
answer_1 = "All plans come with unlimited private models and datasets."
answer_2 = "AutoNLP is an automatic way to train and deploy state-of-the-art NLP models, seamlessly integrated with the Hugging Face ecosystem."
answer_3 = "Based on how much training data and model variants are created, we send you a compute cost and payment link - as low as $10 per job."
answer_4 = "You can host 56 models on HuggingFace"

query_embedding = model.encode(question)
corpus_embedding = model.encode([answer_1,answer_2, answer_3, answer_4])

# print(util.semantic_search(query_embedding, corpus_embedding))

qa_model = pipeline("question-answering")

question ="How many models can I host on HuggingFace?"
question_2= "How old is Gean Franco?"
context = "My name is Gean Franco, I live in London, I'm 24 And I can host 56 models on HuggingFace."

#print(qa_model(question = question_2, context = context))

# ==============================================================================

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus with example sentences
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'Horse is eating grass.',
          'A man is eating pasta.',
          'A Woman is eating Biryani.',
          'The girl is carrying a baby.',
          'The baby is carried by the woman',
          'A man is riding a horse.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.',
          'A cheetah is running behind its prey.',
          'A cheetah chases prey on across a field.',
          'The cheetah is chasing a man who is riding the horse.',
          'man and women with their baby are watching cheetah in zoo'
          ]

corpus_embeddings = embedder.encode(corpus)

#se normalizan los embeddings a la unidad de tamaño
#por buenas prácticas
corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

clustering_model = KMeans(n_clusters=4)
clustering_model.fit(corpus_embeddings)
clustering_assigment = clustering_model.labels_

# print(clustering_assigment)

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(clustering_assigment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id]=[]

    clustered_sentences[cluster_id].append(corpus[sentence_id])
print(clustered_sentences)
