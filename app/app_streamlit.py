# app_streamlit.py
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="RAG Chatbot - OpenAgenda", layout="wide")

st.title(" Chatbot RAG - Événements Tourisme à Lille")

# Champ de saisie
question = st.text_input("Posez votre question :", "")

if st.button("Envoyer") and question.strip():
    with st.spinner("Recherche en cours..."):
        try:
            # Appel à l'API FastAPI locale
            response = requests.post("http://127.0.0.1:8000/ask", json={"question": question})
            response.raise_for_status()
            data = response.json()

            # Affichage de la réponse texte
            st.subheader("Réponse générée par le chatbot :")
            st.text(data["answer"])

            # Affichage des événements récupérés sous forme de tableau
            st.subheader("Événements récupérés :")
            df = pd.DataFrame(data["retrieved_events"])
            st.dataframe(df)

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")
        except Exception as e:
            st.error(f"Erreur interne : {e}")
