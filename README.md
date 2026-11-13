# Retail Product Recommendation System Using SVD

**An end-to-end machine learning recommendation system that uses Singular Value Decomposition (SVD), collaborative filtering, and cosine similarity to generate personalized product recommendations from real-world retail transaction data.**

**Tech Stack:** Python | NumPy | pandas | SciPy | scikit-learn | Matplotlib | Jupyter Notebook

## Overview

Recommendation systems help users discover relevant products by identifying patterns in historical interactions.

This project develops a personalized product recommendation pipeline using the **Online Retail II dataset**, containing more than one million retail transaction records.

The system converts customer purchases into a sparse customer-product interaction matrix, applies truncated SVD to learn lower-dimensional representations, and generates recommendations using latent-factor similarity.

The project also evaluates different SVD ranks, compares SVD with PCA, explores customer segmentation, and measures recommendation performance against a bestseller baseline.

## Key Highlights

- Processed approximately **1.07 million retail transaction records**.
- Constructed a sparse matrix containing **5,474 customers and 4,534 products**.
- Achieved **98.32% matrix sparsity**, enabling efficient storage.
- Reduced matrix storage requirements by **96.61%** using sparse representation.
- Applied truncated SVD using 10, 20, 50, 100, and 200 latent dimensions.
- Achieved **30.32% Hit Rate@10** using the 50-component SVD model.
- Improved Hit Rate@10 by **3.49 percentage points** over the bestseller baseline.
- Compared recommendation performance, reconstruction error, explained variance, computation time, and memory usage.

## Dataset

**Dataset:** Online Retail II  
**Provider:** UCI Machine Learning Repository

**Download:** https://archive.ics.uci.edu/dataset/502/online+retail+ii

The dataset contains transactions from a UK-based online retailer spanning December 2009 to December 2011.

### Dataset Overview

| Property | Value |
|---|---:|
| Original transaction records | 1,067,371 |
| Original columns | 8 |
| Customers after preprocessing | 5,474 |
| Products after preprocessing | 4,534 |
| Customer-product matrix size | 5,474 × 4,534 |
| Non-zero matrix entries | 417,742 |

### Data Preprocessing

The preprocessing pipeline performs the following operations:

1. Combines transactions from the two Excel worksheets.
2. Removes cancelled invoices.
3. Filters transactions with non-positive quantities.
4. Removes transactions with invalid or non-positive prices.
5. Excludes non-product transaction codes.
6. Handles missing customer identifiers and product descriptions.
7. Constructs customer-product purchase interactions.
8. Builds a sparse matrix for dimensionality reduction.
9. Creates a chronological training and testing split for recommendation evaluation.

The original Excel dataset is **not included in this repository**. Users can obtain it directly from the UCI Machine Learning Repository.

## Methodology

### 1. Customer-Product Interaction Matrix

A customer-product interaction matrix is constructed from the cleaned transaction data.

Each row represents a customer, each column represents a product, and the entries describe purchasing interactions.

The matrix contains:

- 5,474 customers
- 4,534 products
- 24,819,116 possible customer-product entries
- 417,742 non-zero entries

Because most customers purchase only a small fraction of the available products, the matrix is highly sparse.

### 2. Sparse Matrix Representation

Sparse storage preserves non-zero interaction values without storing every zero explicitly.

The resulting memory comparison was:

| Representation | Memory |
|---|---:|
| Dense matrix | 94.68 MiB |
| Sparse matrix | 3.21 MiB |
| Memory reduction | 96.61% |

This reduction applies to the interaction matrix's storage representation, not the total memory usage of the application.

### 3. Singular Value Decomposition

Singular Value Decomposition factorizes a matrix into lower-dimensional components:

\[
A = U\Sigma V^T
\]

A truncated rank-\(k\) approximation retains the most important latent dimensions:

\[
A_k = U_k\Sigma_k V_k^T
\]

This allows the system to learn compact representations of purchasing patterns.

The implementation uses scikit-learn's `TruncatedSVD` to calculate latent representations without converting the full sparse interaction matrix into a dense matrix.

### 4. Latent-Factor Recommendation

SVD representations are used to generate ranked recommendations.

The system learns relationships among customers and products through their shared interaction patterns.

Cosine similarity is used to compare product representations.

For two vectors \(a\) and \(b\):

\[
\operatorname{cosine}(a,b)
=
\frac{a \cdot b}{\|a\|_2\|b\|_2}
\]

The system also explores product-to-product similarity recommendations.

### 5. Chronological Train/Test Split

The project uses a time-based split to evaluate recommendation performance on later transactions.

The last two months of transactions are reserved for testing.

Evaluation is restricted to eligible customers and products represented in the training data.

This setup provides a more realistic evaluation than randomly splitting individual purchase interactions across time.

### 6. Recommendation Evaluation

Recommendation quality is measured using **Hit Rate@10**.

A hit occurs when at least one of the top 10 recommended products matches an eligible customer's held-out purchase.

The evaluation compares:

- Bestseller baseline
- TruncatedSVD with cosine similarity
- PCA-based recommendation representations

Multiple dimensionality settings are evaluated to investigate the effect of model complexity.

## Results and Evaluation

### 1. Recommendation Performance

The following results were obtained from the project's recorded evaluation.

| Model | Components | Hit Rate@10 (%) | Customers Hit |
|---|---:|---:|---:|
| Bestseller Baseline | 0 | 26.83 | 514 |
| TruncatedSVD + Cosine | 10 | 27.51 | 527 |
| TruncatedSVD + Cosine | 20 | 29.96 | 574 |
| TruncatedSVD + Cosine | 50 | **30.32** | **581** |
| TruncatedSVD + Cosine | 100 | **30.32** | **581** |
| TruncatedSVD + Cosine | 200 | 27.30 | 523 |
| PCA + Cosine | 50 | **30.32** | **581** |

The 50-component SVD model achieved a Hit Rate@10 of **30.32%**, compared with **26.83%** for the bestseller baseline.

This represents:

- **3.49 percentage points** of absolute improvement.
- Approximately **13.0% relative improvement** over the baseline.

The 50-component SVD model matched the 100-component SVD model in recommendation performance while using fewer latent dimensions.

PCA with 50 components achieved the same recorded Hit Rate@10.

### 2. SVD Rank Comparison

Different SVD ranks were evaluated to understand the relationship between dimensionality and reconstruction quality.

| Components | Explained Variance (%) | Training Time (seconds) | Relative Reconstruction Error |
|---|---:|---:|---:|
| 10 | 15.27 | 0.05 | 0.8989 |
| 20 | 19.34 | 0.08 | 0.8770 |
| 50 | 27.10 | 0.15 | 0.8337 |
| 100 | 35.73 | 0.48 | 0.7828 |
| 200 | 47.91 | 1.29 | 0.7047 |

Increasing the number of components improved explained variance and reduced reconstruction error.

However, the highest reconstruction quality did not necessarily produce the best recommendation accuracy.

For example, the 200-component model achieved lower reconstruction error than the 50-component model but a lower Hit Rate@10.

This illustrates the distinction between reconstructing an interaction matrix and predicting relevant products.

### 3. PCA vs TruncatedSVD

A 50-component comparison was performed between PCA and TruncatedSVD.

| Metric | TruncatedSVD | PCA |
|---|---:|---:|
| Components | 50 | 50 |
| Fit time (seconds) | 0.1893 | 0.2675 |
| Python peak memory (MB) | 4.91 | 99.53 |
| Stored factors (MB) | 1.91 | 1.93 |
| Explained variance (%) | 27.10 | 27.17 |
| Hit Rate@10 (%) | 30.32 | 30.32 |

In this recorded experiment, TruncatedSVD required less measured fitting time and Python-tracked peak memory than PCA.

Both approaches achieved similar explained variance and identical recorded recommendation performance at 50 components.

These measurements reflect one experimental run and are not universal performance benchmarks.

### 4. Customer Segmentation

The project also explores customer segmentation using lower-dimensional purchasing representations.

Customer embeddings are analyzed and visualized to investigate similarities among purchasing behaviors.

The segmentation component demonstrates how latent-factor representations can support exploratory customer analysis beyond product recommendation.

## Project Structure

```text
Retail-Recommendation-System-SVD/
├── src/
│   ├── build_matrix.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── compare_models.py
│   ├── recommend_cosine.py
│   └── customer_map.py
│
├── notebooks/
│   └── project_analysis.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

### Main Components

| File | Description |
|---|---|
| `build_matrix.py` | Data loading, cleaning, chronological splitting, and sparse matrix construction |
| `train_model.py` | TruncatedSVD training and latent-factor generation |
| `evaluate.py` | Recommendation evaluation and comparison with baselines |
| `compare_models.py` | PCA and TruncatedSVD comparison |
| `recommend_cosine.py` | Product similarity and cosine-based recommendations |
| `customer_map.py` | Customer representation analysis and visualization |
| `project_analysis.ipynb` | Exploratory analysis, results interpretation, and visualizations |

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Venkat18-bit/Retail-Recommendation-System-SVD.git
cd Retail-Recommendation-System-SVD
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the Dataset

Download the Online Retail II dataset from:

https://archive.ics.uci.edu/dataset/502/online+retail+ii

Create the following directory in the project root:

```text
data/raw/
```

Save the downloaded Excel workbook as:

```text
data/raw/online_retail_II.xlsx
```

The downloaded dataset must contain both original transaction worksheets.

### 5. Run the Project

Run the following commands sequentially from the repository root.

**Step 1 — Build the customer-product matrix**

```bash
python src/build_matrix.py
```

**Step 2 — Train SVD models**

```bash
python src/train_model.py
```

**Step 3 — Compare PCA and SVD**

```bash
python src/compare_models.py
```

**Step 4 — Evaluate recommendation performance**

```bash
python src/evaluate.py
```

**Step 5 — Generate cosine similarity recommendations**

```bash
python src/recommend_cosine.py
```

**Step 6 — Analyze customer segments**

```bash
python src/customer_map.py
```

**Step 7 — Open the analysis notebook**

Open `notebooks/project_analysis.ipynb` in VS Code or Jupyter.

The notebook reads locally generated artifacts from `outputs/`.

The raw dataset, customer identifiers, and generated model files are excluded from GitHub.

## Privacy and Data Handling

This repository intentionally excludes:

- Raw retail transaction datasets
- Generated customer-level purchase records
- Customer identifiers and mapping arrays
- Saved model artifacts
- Local virtual environments
- Environment variable files and credentials
- Personal academic submission documents

The `.gitignore` file prevents matching untracked files from being added to Git.

Users can reproduce the analysis by obtaining the original dataset and generating the required outputs locally.

## Limitations

### Data Sparsity

Most customers interact with relatively few products, limiting the available evidence for personalized recommendations.

### Cold-Start Problem

New customers and new products may not have sufficient interaction history to support reliable recommendations.

### Implicit Feedback

Purchasing history provides implicit feedback rather than explicit product ratings.

A purchase does not necessarily imply long-term satisfaction or preference.

### Model Evaluation

Hit Rate@10 measures whether at least one relevant product appears among the top 10 recommendations.

It does not fully capture ranking quality, recommendation diversity, novelty, or business impact.

### Experimental Performance

Timing and memory measurements depend on hardware, software versions, dataset characteristics, and implementation choices.

## Future Improvements

Potential extensions include:

- Precision@K and Recall@K evaluation
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Hybrid content-based and collaborative filtering
- Implicit-feedback matrix factorization
- Cold-start recommendation strategies
- Interactive recommendation dashboard
- Additional recommendation diversity analysis

## Conclusion

This project demonstrates the use of linear algebra and machine learning to develop a personalized recommendation system from real-world retail transactions.

It combines data preprocessing, sparse matrix construction, dimensionality reduction, latent-factor modeling, recommendation evaluation, and exploratory customer analysis.

The results show that truncated SVD can provide meaningful improvements over a simple bestseller baseline while retaining a compact latent representation.

The project also highlights an important lesson in recommendation systems: **better matrix reconstruction does not always mean better recommendation performance.**