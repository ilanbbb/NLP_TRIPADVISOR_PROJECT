#  TripAdvisor Place Retrieval — NLP Project

> Projet NLP — Information Retrieval sur données TripAdvisor  
> Par **Boulmier Ilan**

---

## Description

Ce projet implémente un système de **recherche et recommandation de lieux** (hôtels, restaurants, attractions) basé sur les avis TripAdvisor. À partir d'un avis client, le système retrouve les lieux les plus similaires dans la base de données en utilisant différentes approches NLP.

L'évaluation se fait à deux niveaux :
- **Level 1** : le lieu recommandé est-il du même type (H/R/A/AP) ?
- **Level 2** : le lieu recommandé partage-t-il les mêmes métadonnées (type de cuisine, gamme de prix, catégorie d'activité...) ?

---
##  MON RAPPORT

```
# Lancer le pdf
DIA2_report_nlp.pdf
```

## Méthodologie

### 1. Prétraitement
- Filtrage des avis anglais (`langue == "en"`)
- Concaténation de tous les avis par lieu (`idplace`)
- Jointure avec les métadonnées des lieux

### 2. Modèles testés

#### BM25 (Baseline)
Modèle probabiliste de recherche d'information basé sur la fréquence des termes. Implémenté avec `rank-bm25`.

#### TF-IDF + Similarité Cosinus
Vectorisation TF-IDF sur les avis concaténés, puis recherche par similarité cosinus entre la requête et les documents du corpus test.

#### TF-IDF Amélioré
Même approche avec paramètres optimisés :
- `ngram_range=(1,2)` — unigrammes + bigrammes
- `max_features=20000`
- `min_df=2`, `max_df=0.8`
- Suppression des stopwords anglais

#### Sentence Transformers (Embeddings sémantiques)
Modèle `all-MiniLM-L6-v2` de HuggingFace pour des embeddings contextuels profonds.

---

##  Installation

```bash
pip install pandas numpy scikit-learn rank-bm25 sentence-transformers tqdm
```

---

##  Utilisation

```python
# Lancer le notebook
jupyter notebook Projet_NLP.ipynb
```

Le notebook est structuré en 3 parties :
1. **Data Cleaning** — filtrage, concaténation, split train/test
2. **Model + Evaluation** — BM25, TF-IDF, améliorations, Sentence Transformers
3. **Conclusion** — tableau comparatif des résultats

## Auteur 
Etudiant - BOULMIER Ilan

