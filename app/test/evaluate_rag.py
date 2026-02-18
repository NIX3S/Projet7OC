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
import pandas as pd
import asyncio
from ragas.metrics.collections import ExactMatch, NonLLMStringSimilarity, DistanceMeasure

# On renomme les colonnes pour correspondre à l’API RAGas
eval_df.rename(
    columns={"gold_answer": "ground_truth", "generated_answer": "answer"},
    inplace=True
)

# Instanciation des métriques
exact_match_scorer = ExactMatch()
similarity_scorer = NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN)

# Fonction async pour évaluer toute la DataFrame
async def evaluate_df(df):
    exact_scores = []
    similarity_scores = []
    coverage_scores = []

    for idx, row in df.iterrows():
        ref = row["ground_truth"]
        pred = row["answer"]

        # Exact Match
        em = await exact_match_scorer.ascore(reference=ref, response=pred)
        exact_scores.append(em.value)

        # Similarité NonLLM (Levenshtein)
        sim = await similarity_scorer.ascore(reference=ref, response=pred)
        similarity_scores.append(sim.value)

        # Coverage approximatif : proportion de mots du gold_answer présents dans la réponse
        ref_words = set(ref.split())
        pred_words = set(pred.split())
        coverage = len(ref_words & pred_words) / len(ref_words) if ref_words else 0
        coverage_scores.append(coverage)

    return exact_scores, similarity_scores, coverage_scores

# Exécuter la boucle async
exact_scores, similarity_scores, coverage_scores = asyncio.run(evaluate_df(eval_df))

# Ajouter les métriques au DataFrame
eval_df["exact_match"] = exact_scores
eval_df["similarity_score"] = similarity_scores
eval_df["coverage"] = coverage_scores

# Affichage des résultats globaux (moyennes)
print("\n=== Résultats globaux ===")
print(f"Exact Match moyen      : {eval_df['exact_match'].mean():.3f}")
print(f"Similarity Score moyen : {eval_df['similarity_score'].mean():.3f}")
print(f"Coverage moyen         : {eval_df['coverage'].mean():.3f}")

# Sauvegarder le rapport détaillé
eval_df.to_csv("rag_evaluation_report.csv", index=False)
print("\nRapport détaillé sauvegardé dans 'rag_evaluation_report.csv'")
