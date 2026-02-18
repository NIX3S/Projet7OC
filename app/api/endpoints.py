from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from app.rag.rag_system import rag_system
from app.rag.rebuild_faiss import rebuild_faiss_index  
from dotenv import load_dotenv
import os
import time
from datetime import datetime, timezone
from fastapi import FastAPI, APIRouter, HTTPException, Header
app = FastAPI(title="RAG API", version="1.0.0")

try :
    load_dotenv()
except:
    None
API_KEY = os.getenv("ADMIN_API_KEY")
router = APIRouter()
# --- Enregistrement du temps de démarrage ---
START_TIME = time.time()

class QuestionRequest(BaseModel):
    question: str

@router.get("/metadata")
def get_metadata():
    """
    Retourne les informations générales sur l’API.
    """
    return {
        "name": "RAG Event Assistant",
        "version": app.version,
        "description": "API RAG permettant d'interroger une base d'événements indexée avec FAISS",
        "status": "POC-ready"
    }


@router.get("/health")
def health_check():
    """
    Vérifie l’état du service et retourne  l'uptime.
    """
    uptime_seconds = int(time.time() - START_TIME)

    return {
        "status": "ok",
        "uptime_seconds": uptime_seconds,
        "uptime_human": str(datetime.now(timezone.utc) - datetime.fromtimestamp(START_TIME, timezone.utc))
    }

@router.post("/ask")
def ask_question(req: QuestionRequest):
    try:
        return rag_system.ask(req.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rebuild")
def rebuild_vectorstore(x_api_key: str = Header(None)):
    """
    Endpoint pour reconstruire la base vectorielle FAISS à la demande.
    Télécharge les événements, génère embeddings, construit l'index et sauvegarde.
    """
    #print(x_api_key)
    #print(API_KEY)
    if x_api_key == API_KEY:
        try:
            total_events, total_indexed = rebuild_faiss_index()
            # Recharger l’index dans le RAG system déjà instancié
            rag_system.load_vectorstore()
            return {
                "status": "success",
                "message": f"Rebuild terminé : {total_events} événements téléchargés, {total_indexed} vecteurs indexés."
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        return {"status": "error", "message": "Invalid Key"}