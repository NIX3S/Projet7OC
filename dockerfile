# Étape 1 : image Python officielle
FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers principaux
COPY requirements.txt .
COPY .env .
COPY faiss_index_openagenda.idx .
COPY metadata_openagenda.pkl .
COPY app/ ./app/

# Installer les dépendances
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Exposer le port de l'API
EXPOSE 8000

# Commande par défaut : rebuild FAISS + lancer l'API
CMD ["bash", "-c", "python app/rag/rebuild_faiss.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
