# BookTrends RecSys Spotlight Script

## Introduction (0:00 - 0:45)
- Welcome audience and introduce the BookTrends recommendation project.
- Highlight motivation: helping readers navigate large catalogs via fair, personalized suggestions.

## Data & Pipeline (0:45 - 1:30)
- Describe Goodreads-derived datasets and preprocessing workflow.
- Explain automated Makefile steps: data prep, ALS training, evaluation, and parity assessment.
- Mention reproducibility via fixed seed and versioned artifacts.

## Model Insights (1:30 - 2:15)
- Share key metrics (Precision@10, Recall@10, NDCG@10) from evaluation.
- Discuss selected hyperparameters and what they imply about collaborative filtering trade-offs.
- Present notable EDA visuals: rating distribution, genre popularity, user-item interaction heatmap.

## Responsible AI & Parity (2:15 - 2:45)
- Summarize fairness findings from parity analysis across user regions and book genres.
- Address mitigation ideas and roadmap for continuous monitoring.

## Demo & Call to Action (2:45 - 3:00)
- Preview Streamlit demo experience highlighting recommendation flow.
- Invite feedback and collaboration for future releases.
