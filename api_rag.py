# api_rag.py
import faiss
import pickle
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI
from mistralai import Mistral
import numpy as np

# --- Charger l'environnement ---
load_dotenv()

# --- Schéma pour la requête POST /ask ---
class QuestionRequest(BaseModel):
    question: str

# --- RAG System encapsulé dans une classe ---
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
        # Client embeddings
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
        # Cette fonction peut être étendue pour recalculer FAISS
        self.load_vectorstore()
        return "Base vectorielle reconstruite avec succès."

    def ask(self, question: str, k: int = 5):
        if not question.strip():
            raise ValueError("Question vide.")
        # Générer le vecteur de la question
        response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=[question]
        )
        query_embedding = response.data[0].embedding
        query_vector = np.array([query_embedding], dtype='float32')  # shape = (1, dim)

        # Recherche top-k via FAISS
        distances, indices = self.index.search(query_vector, k)
        retrieved_docs = [self.docstore_dict[self.index_to_docstore_id[i]] for i in indices[0]]

        # Éliminer doublons
        unique_docs, seen_titles = [], set()
        for doc in retrieved_docs:
            title = doc.metadata.get("title")
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)
        retrieved_docs = unique_docs[:k]

        # Construire le contexte
        context_lines = []
        for doc in retrieved_docs:
            meta = doc.metadata
            line = f"{meta.get('title')} | {meta.get('start')} | {meta.get('location_city')}"
            context_lines.append(line)
        context_text = "\n".join(context_lines)

        # Construire le prompt
        prompt = f"""Réponds à la question suivante uniquement avec la liste d'événements présents dans le contexte.
Ne réécris pas les titres, ne change pas les dates ni les lieux.
Chaque événement doit être sur une ligne : Titre - Date - Lieu
Si CONTEXTE est Vide dans ce cas tu réponds AUCUN Evenement de ce type
CONTEXTE :
{context_text}

QUESTION : {question}

Réponse précise et concise :"""

        # Appel LLM
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

# --- Initialiser FastAPI et le système RAG ---
app = FastAPI(title="RAG API - OpenAgenda + Mistral")
rag_system = RAGSystem()

# --- Endpoint pour poser une question ---
@app.post("/ask")
def ask_question(req: QuestionRequest):
    try:
        return rag_system.ask(req.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint pour reconstruire la base vectorielle ---
@app.post("/rebuild")
def rebuild():
    try:
        message = rag_system.rebuild_vectorstore()
        return {"status": "success", "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
