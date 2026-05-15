"""
RAGAS Evaluation — continuous RAG quality measurement.

Usage:
    python -m evals.ragas_eval

Requires:
    pip install ragas

Evaluates faithfulness, answer_relevancy, and context_precision
on a golden dataset. To be scheduled in CI (nightly).
"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision


GOLDEN_DATASET = {
    "question": [
        "Qu'est-ce que le RAG dans le contexte des télécoms ?",
        "Comment déployer une antenne 5G ?",
    ],
    "answer": [
        "Le RAG (Retrieval-Augmented Generation) est une technique qui combine recherche documentaire et génération de texte pour répondre à des questions sur des documents techniques télécoms.",
        "Le déploiement d'une antenne 5G nécessite une étude de site, une analyse de couverture, et l'installation des équipements radio.",
    ],
    "contexts": [
        ["Le RAG est utilisé pour la recherche documentaire dans les télécoms."],
        ["Le déploiement 5G inclut l'étude de site et l'installation d'antennes."],
    ],
}


def run_evaluation():
    dataset = Dataset.from_dict(GOLDEN_DATASET)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    print("RAGAS Evaluation Results:")
    for metric, score in result.items():
        print(f"  {metric}: {score:.4f}")
    return result


if __name__ == "__main__":
    run_evaluation()
