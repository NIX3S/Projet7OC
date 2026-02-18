import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from mistralai import Mistral
import numpy as np
import faiss
import pickle
import time
import os
from dotenv import load_dotenv

try :
    load_dotenv()
except:
    None
API_KEY = os.getenv("MISTRAL_API_KEY")

def rebuild_faiss_index():
    """
    Rebuild the FAISS index from OpenAgenda events and save the index + metadata.
    """
    BASE_URL = "https://hub.huwise.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records/"
    limit = 100
    offset = 0
    all_events = []
    date_un_an = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # --- Télécharger les événements ---
    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "where": f'location_department="Nord" AND firstdate_begin >= "{date_un_an}"'
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        events = data.get("results", [])
        if not events:
            break
        all_events.extend(events)
        offset += limit
        print(f"Page offset {offset} → {len(events)} événements récupérés")

    print(f"Total événements récupérés : {len(all_events)}")

    # --- Créer DataFrame et nettoyer ---
    df = pd.DataFrame(all_events)
    df = df[[
        "uid", "title_fr", "description_fr",
        "firstdate_begin", "firstdate_end",
        "location_name", "location_city",
        "location_department", "keywords_fr"
    ]].copy()
    df.rename(columns={
        "uid": "id",
        "title_fr": "title",
        "description_fr": "description",
        "firstdate_begin": "start",
        "firstdate_end": "end"
    }, inplace=True)

    def clean_html(text):
        if pd.isna(text):
            return ""
        return BeautifulSoup(text, "html.parser").get_text()

    df["description"] = df["description"].apply(clean_html)
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df.drop_duplicates(subset="id", inplace=True)
    df["start"] = df["start"].dt.strftime("%d %B %Y")
    df["end"] = df["end"].dt.strftime("%d %B %Y")
    df["keywords_str"] = df["keywords_fr"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    df["text_for_vector"] = (
        df["title"].fillna("") + " | " +
        df["description"].fillna("") + " | " +
        df["location_city"].fillna("") + " | " +
        df["keywords_str"].fillna("") + " | " +
        df["location_name"].fillna("") + " | " +
        df["start"] + " au " + df["end"]
    )

    # --- Générer embeddings Mistral ---
    client = Mistral(api_key=API_KEY)
    BATCH_SIZE = 128
    embeddings = []
    texts = df["text_for_vector"].tolist()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        response = client.embeddings.create(model="mistral-embed", inputs=batch)
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
        print(f"Batch {i//BATCH_SIZE + 1} traité")
        time.sleep(0.5)
    df["embedding"] = embeddings

    # --- Créer index FAISS ---
    embeddings_matrix = np.vstack(df["embedding"].values).astype('float32')
    embedding_dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_matrix)

    # --- Sauvegarder index et métadonnées ---
    faiss.write_index(index, "faiss_index_openagenda.idx")
    df_metadata = df[["embedding", "title", "description", "start", "end", 
                      "location_name", "location_city", "location_department", "keywords_fr"]].copy()
    with open("metadata_openagenda.pkl", "wb") as f:
        pickle.dump(df_metadata.to_dict(orient="records"), f)

    print("Rebuild FAISS terminé avec succès.")
    return len(df), index.ntotal
