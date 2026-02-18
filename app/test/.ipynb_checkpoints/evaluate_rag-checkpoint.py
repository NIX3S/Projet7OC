import pandas as pd
from ragas import evaluate
import requests

API_URL = "http://127.0.0.1:8000/ask"  # ton endpoint RAG
GOLD_FILE = "gold_answers.csv"

# Charger le dataset de test
df = pd.read_csv(GOLD_FILE, sep=";")  # adapte le séparateur si besoin

results = []

for idx, row in df.iterrows():
    question = row["question"]
    gold_answer = row["answer"]

    # Appel à l'API RAG
    response = requests.post(API_URL, json={"question": question})
    if response.status_code != 200:
        print(f"Erreur pour la question {idx}: {response.text}")
        continue

    generated_answer = response.json()["answer"]

    results.append({
        "question": question,
        "gold_answer": gold_answer,
        "generated_answer": generated_answer
    })

eval_df = pd.DataFrame(results)

# -----------------------------
# Évaluation simplifiée RAGas
# -----------------------------
# RAGas permet de passer un DataFrame avec colonnes ['question', 'answer', 'ground_truth']
# On renomme juste pour coller à l’API RAGas
eval_df.rename(columns={"gold_answer": "ground_truth", "generated_answer": "answer"}, inplace=True)

metrics = evaluate(
    eval_df,  # passe le DataFrame complet
    metrics=["exact_match", "string_similarity", "coverage"],
    return_executor=False  # exécute directement
)

print("\n=== Résultats globaux ===")
for k, v in metrics.items():
    if k != "per_example":
        print(f"{k}: {v:.3f}")

# Ajouter les métriques par question
eval_df["exact_match"] = [m["exact_match"] for m in metrics["per_example"]]
eval_df["string_similarity"] = [m["string_similarity"] for m in metrics["per_example"]]
eval_df["coverage"] = [m["coverage"] for m in metrics["per_example"]]

# Sauvegarder un rapport détaillé
eval_df.to_csv("rag_evaluation_report.csv", index=False)
print("\nRapport détaillé sauvegardé dans 'rag_evaluation_report.csv'")
