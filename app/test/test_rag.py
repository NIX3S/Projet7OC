import pytest
import pandas as pd
import asyncio
import os
import subprocess
from ragas.metrics.collections import NonLLMStringSimilarity, DistanceMeasure

# Fichier CSV pré-généré contenant les réponses déjà récupérées
EVAL_CSV = "rag_evaluation_report.csv"
EVAL_SCRIPT = "evaluate_rag.py"

@pytest.fixture(scope="session")
def scored_df():
    """
    Charge le CSV d'évaluation.
    S'il n'existe pas, lance le script de génération automatiquement.
    """

    # Si le CSV n'existe pas → on le génère
    if not os.path.exists(EVAL_CSV):
        print("Fichier d'évaluation absent. Génération en cours...")
        subprocess.run(["python", EVAL_SCRIPT], check=True)

    df = pd.read_csv(EVAL_CSV)

    # Sécurité : recalcul si colonnes manquantes
    if "similarity_score" not in df.columns or "coverage" not in df.columns:
        similarity_scorer = NonLLMStringSimilarity(
            distance_measure=DistanceMeasure.LEVENSHTEIN
        )

        async def _compute_scores(df):
            similarity_scores = []
            coverage_scores = []

            for _, row in df.iterrows():
                ref = row["ground_truth"]
                pred = row["answer"]

                sim = await similarity_scorer.ascore(
                    reference=ref,
                    response=pred
                )
                similarity_scores.append(sim.value)

                ref_words = set(ref.split())
                pred_words = set(pred.split())
                coverage = (
                    len(ref_words & pred_words) / len(ref_words)
                    if ref_words else 0
                )
                coverage_scores.append(coverage)

            df["similarity_score"] = similarity_scores
            df["coverage"] = coverage_scores
            return df

        df = asyncio.run(_compute_scores(df))

    return df

def test_similarity_scores(scored_df):
    """Vérifie que tous les similarity scores sont entre 0 et 1."""
    assert all(0 <= s <= 1 for s in scored_df["similarity_score"]), "Similarity score hors bornes 0-1"

def test_coverage_scores(scored_df):
    """Vérifie que tous les coverage scores sont entre 0 et 1."""
    assert all(0 <= c <= 1 for c in scored_df["coverage"]), "Coverage score hors bornes 0-1"

def test_answers_not_empty(scored_df):
    """Vérifie que toutes les réponses générées sont des chaînes non vides."""
    assert all(isinstance(a, str) and a.strip() != "" for a in scored_df["answer"]), "Réponse vide ou non string"
