import faiss
import pickle
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI
from mistralai import Mistral
import numpy as np

try :
    load_dotenv()
except:
    None

class RAGSystem:
    def __init__(self):
        self.index_file = "faiss_index_openagenda.idx"
        self.metadata_file = "metadata_openagenda.pkl"
        self.load_vectorstore()
        # LLM
        self.llm = ChatMistralAI(
            model="mistral-large-latest",
            api_key=os.getenv("MISTRAL_API_KEY")
        )
        # Embeddings
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    def load_vectorstore(self):
        print("Chargement FAISS et métadonnées...")
        self.index = faiss.read_index(self.index_file)
        with open(self.metadata_file, "rb") as f:
            metadata = pickle.load(f)
        self.docs = [Document(page_content=d["title"] + " | " + d.get("description", ""), metadata=d) for d in metadata]
        self.docstore_dict = {i: doc for i, doc in enumerate(self.docs)}
        self.index_to_docstore_id = {i: i for i in range(len(self.docs))}

    def rebuild_vectorstore(self):
        self.load_vectorstore()
        return "Base vectorielle reconstruite avec succès."

    def ask(self, question: str, k: int = 100):
        if not question.strip():
            raise ValueError("Question vide.")

        # --- 1. Générer le vecteur de la question ---
        response = self.client.embeddings.create(model="mistral-embed", inputs=[question])
        query_embedding = response.data[0].embedding
        query_vector = np.array([query_embedding], dtype='float32')

        # --- 2. Top-k via FAISS ---
        distances, indices = self.index.search(query_vector, k)
        retrieved_docs = [self.docstore_dict[self.index_to_docstore_id[i]] for i in indices[0]]

        # --- 3. Éliminer doublons ---
        unique_docs, seen_titles = [], set()
        for doc in retrieved_docs:
            title = doc.metadata.get("title")
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)
        retrieved_docs = unique_docs

        # --- 4. Trier par date (du plus proche au plus ancien) ---
        from datetime import datetime
        today = datetime.today()
        
        def parse_date(date_str):
            try:
                return datetime.strptime(date_str, "%d %B %Y")
            except:
                return datetime.min  # mettre les dates inconnues à très loin dans le passé

        retrieved_docs.sort(key=lambda d: parse_date(d.metadata.get("start", "")))

        # --- 5. Limiter à k documents après tri ---
        retrieved_docs = retrieved_docs[:k]

        # --- 6. Construire le contexte pour le LLM ---
        context_lines = []
        for doc in retrieved_docs:
            meta = doc.metadata
            line = f"{meta.get('title')} | {meta.get('start')} | {meta.get('location_city')}"
            context_lines.append(line)
        context_text = "\n".join(context_lines)

        # Prompt LLM

        prompt = f"""Réponds uniquement avec les événements pertinents du contexte fourni.

        CONTEXTE :
        {context_text}

        QUESTION : {question}

        Chaque événement doit être sur une ligne : Titre - Date - Lieu
        Si aucun événement ne correspond à la demande, répond AUCUN Evenement

        Aujourd'hui : {today.strftime('%d %B %Y')}

        Instructions supplémentaires :
        - Si la question mentionne "à venir", inclut uniquement les événements dont la date est >= aujourd'hui.
        - Si la question mentionne "il y a", inclut uniquement les événements dont la date est < aujourd'hui
        - Ne réécris pas les titres, ne change pas les dates ni les lieux.
        - Réponse précise et concise.

        Réponse :"""

        answer = self.llm.invoke(prompt)
        return {
            "answer": answer.content if hasattr(answer, 'content') else answer,
            "retrieved_events": [
                {
                    "title": doc.metadata.get("title"),
                    "city": doc.metadata.get("location_city"),
                    "start": doc.metadata.get("start"),
                    "end": doc.metadata.get("end")
                }
                for doc in retrieved_docs
            ]
        }

# Instancier globalement pour réutilisation dans endpoints
rag_system = RAGSystem()
