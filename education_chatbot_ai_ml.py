"""
Education Chatbot — AI/ML Concepts (15 fixed Q&A)
-------------------------------------------------
- Meets requirement: answers 15 predefined questions correctly.
- Bonus: handles small phrasing variations via fuzzy matching & aliases.
- Tech stack: Python 3.x + (optional) Streamlit for UI. CLI mode also included.

Run (UI):
  pip install streamlit
  streamlit run education_chatbot_ai_ml.py

Run (CLI):
  python education_chatbot_ai_ml.py --cli

Files needed: just this single .py file.
"""
from __future__ import annotations

import re
import sys
import difflib
from typing import Dict, List, Tuple

# 1) Knowledge base: 15 Q&A

QA_ITEMS: List[Dict] = [
    {
        "q": "What is Artificial Intelligence?",
        "a": (
            "Artificial Intelligence (AI) is the field of building systems that perform tasks "
            "requiring human-like intelligence—such as perception, reasoning, learning, and decision-making."
        ),
        "aliases": ["Define AI", "Explain Artificial Intelligence", "Meaning of AI"],
    },
    {
        "q": "What is Machine Learning?",
        "a": (
            "Machine Learning (ML) is a subset of AI where models learn patterns from data to make predictions or decisions "
            "without being explicitly programmed for each rule."
        ),
        "aliases": ["Define Machine Learning", "Explain ML", "Meaning of ML"],
    },
    {
        "q": "What is the difference between AI and ML?",
        "a": (
            "AI is the broader goal of making machines intelligent; ML is a subset of AI focused on learning from data. "
            "All ML is AI, but not all AI is ML."
        ),
        "aliases": ["AI vs ML", "Difference AI and ML", "How is ML different from AI"],
    },
    {
        "q": "What is supervised learning?",
        "a": (
            "Supervised learning trains models on labeled data (inputs with known targets) to predict labels for new inputs."
        ),
        "aliases": ["Define supervised learning", "Explain supervised learning", "Supervised learning definition"],
    },
    {
        "q": "What is unsupervised learning?",
        "a": (
            "Unsupervised learning finds structure in unlabeled data, e.g., clustering and dimensionality reduction."
        ),
        "aliases": ["Define unsupervised learning", "Explain unsupervised", "Unsupervised learning definition"],
    },
    {
        "q": "What is overfitting?",
        "a": (
            "Overfitting happens when a model learns noise or spurious patterns in the training data, "
            "leading to poor generalization on new data."
        ),
        "aliases": ["Define overfitting", "Explain overfitting"],
    },
    {
        "q": "What is underfitting?",
        "a": (
            "Underfitting occurs when a model is too simple to capture the underlying pattern, resulting in high error on both "
            "training and test data."
        ),
        "aliases": ["Define underfitting", "Explain underfitting"],
    },
    {
        "q": "What is the bias-variance trade-off?",
        "a": (
            "The bias-variance trade-off balances underfitting (high bias) and overfitting (high variance) to minimize total error."
        ),
        "aliases": ["Explain bias variance tradeoff", "Bias variance trade off", "bias vs variance"],
    },
    {
        "q": "What is a neural network?",
        "a": (
            "A neural network is a model composed of layers of interconnected nodes (neurons) that learn representations by "
            "applying weighted sums and nonlinear activation functions."
        ),
        "aliases": ["Define neural network", "Explain ANN", "What is ANN"],
    },
    {
        "q": "How do we split data into train/validation/test?",
        "a": (
            "Common practice is to split data into training (fit the model), validation (tune hyperparameters), and test (final unbiased evaluation), "
            "e.g., 70/15/15 or 80/10/10 depending on dataset size."
        ),
        "aliases": ["Dataset split", "Train validation test split", "How to split data"],
    },
    {
        "q": "Explain accuracy, precision, and recall.",
        "a": (
            "Accuracy is overall correctness. Precision is the fraction of predicted positives that are truly positive. "
            "Recall is the fraction of true positives that were correctly identified."
        ),
        "aliases": ["Accuracy vs precision vs recall", "Define precision and recall", "What is accuracy precision recall"],
    },
    {
        "q": "What is a confusion matrix?",
        "a": (
            "A confusion matrix is a table for classification showing counts of TP, FP, FN, and TN, helping compute metrics like precision and recall."
        ),
        "aliases": ["Define confusion matrix", "Explain confusion matrix"],
    },
    {
        "q": "What is gradient descent?",
        "a": (
            "Gradient descent is an optimization algorithm that iteratively updates parameters in the direction opposite the gradient to minimize a loss function."
        ),
        "aliases": ["Define gradient descent", "Explain GD", "How does gradient descent work"],
    },
    {
        "q": "What is feature scaling and why is it important?",
        "a": (
            "Feature scaling normalizes or standardizes input features so they share a comparable range, which helps algorithms like gradient descent converge faster and models like KNN/SVM perform better."
        ),
        "aliases": ["Define feature scaling", "Normalization vs standardization", "Why scale features"],
    },
    {
        "q": "What is cross-validation?",
        "a": (
            "Cross-validation (e.g., k-fold) splits the training data into k folds, training on k−1 folds and validating on the remaining fold, rotating to estimate generalization reliably."
        ),
        "aliases": ["Define cross validation", "Explain k-fold CV", "What is kfold"],
    },
]

# We have exactly 15 items
assert len(QA_ITEMS) == 15, "There must be exactly 15 predefined questions."

# 2) Normalization & fuzzy matching helpers


def normalize(text: str) -> str:
    text = text.strip().lower()
    # Remove punctuation & extra spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# Build a lookup of canonical question -> data
CANONICAL: Dict[str, Dict] = {}
ALL_PHRASES: List[str] = []
for item in QA_ITEMS:
    canon = normalize(item["q"])  # canonical normalized
    CANONICAL[canon] = item
    ALL_PHRASES.append(canon)
    for alias in item.get("aliases", []):
        ALL_PHRASES.append(normalize(alias))

# Map each alias phrase to its canonical question (by best fuzzy match among canonical questions)
ALIAS_TO_CANON: Dict[str, str] = {}
for item in QA_ITEMS:
    canon = normalize(item["q"])
    for alias in item.get("aliases", []):
        alias_n = normalize(alias)
        ALIAS_TO_CANON[alias_n] = canon

# 3) Core answer function


def answer_question(user_q: str) -> Tuple[str, str]:
    """
    Returns (best_matched_question, answer).
    Matching strategy hierarchy:
      1) Exact match against canonical questions (normalized)
      2) Exact match against known aliases
      3) Fuzzy match (difflib) against all phrases; if close enough, map to canonical
      4) Otherwise: helpful fallback with suggested close questions
    """
    nq = normalize(user_q)

    # 1) Exact canonical match
    if nq in CANONICAL:
        item = CANONICAL[nq]
        return (item["q"], item["a"])

    # 2) Exact alias match
    if nq in ALIAS_TO_CANON:
        canon = ALIAS_TO_CANON[nq]
        item = CANONICAL[canon]
        return (item["q"], item["a"])

    # 3) Fuzzy: compare to ALL_PHRASES
    # get_close_matches returns best matches by similarity ratio
    candidates = difflib.get_close_matches(nq, ALL_PHRASES, n=3, cutoff=0.6)
    if candidates:
        # If candidate is an alias, map to its canonical
        best = candidates[0]
        canon = ALIAS_TO_CANON.get(best, best)
        if canon in CANONICAL:
            item = CANONICAL[canon]
            return (item["q"], item["a"])

    # 4) Fallback with suggestions
    suggestions = difflib.get_close_matches(nq, [normalize(x["q"]) for x in QA_ITEMS], n=3, cutoff=0.4)
    if suggestions:
        readable = [CANONICAL[s]["q"] for s in suggestions if s in CANONICAL]
        hint = "\nDid you mean: " + "; ".join(readable)
    else:
        hint = "\nTry asking one of the 15 supported questions (see list)."
    return ("No exact match", "I'm set up to answer 15 specific AI/ML questions. "
            "Please try a supported question or a close variation." + hint)

# 4) Streamlit UI 


def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="Education Chatbot — AI/ML", page_icon="🤖")
    st.title("🤖 Education Chatbot — AI/ML Concepts")
    st.caption("Answers 15 predefined questions. Handles simple phrasing variations.")

    with st.expander("See all 15 supported questions"):
        for i, item in enumerate(QA_ITEMS, start=1):
            st.markdown(f"**{i}. {item['q']}**")

    st.divider()
    q = st.text_input("Ask a question", placeholder="e.g., What is Machine Learning?")
    if st.button("Ask") or q:
        best_q, ans = answer_question(q)
        if best_q == "No exact match":
            st.error(ans)
        else:
            st.success(f"Matched: {best_q}")
            st.write(ans)

    st.divider()
    st.subheader("Evaluator Helper")
    if st.button("Run all 15 demo Q&A"):
        for item in QA_ITEMS:
            st.markdown(f"**Q:** {item['q']}")
            st.write(f"**A:** {item['a']}")
            st.markdown("---")

# 5) CLI mode


def run_cli():
    print("Education Chatbot — AI/ML (15 questions)")
    print("Type your question (or 'list' to show questions, 'quit' to exit).\n")
    while True:
        try:
            user_q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print() 
            break
        if not user_q:
            continue
        if user_q.lower() in {"quit", "exit"}:
            break
        if user_q.lower() in {"list", "help"}:
            for i, item in enumerate(QA_ITEMS, start=1):
                print(f"{i}. {item['q']}")
            continue
        best_q, ans = answer_question(user_q)
        if best_q == "No exact match":
            print(f"Bot: {ans}\n")
        else:
            print(f"Bot (matched '{best_q}'): {ans}\n")

# 6) Entrypoint

if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli()
    else:
        run_streamlit()

