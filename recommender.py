# DecodeLabs Project 3

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ingestion
print("=" * 45)
print("   DecodeLabs — Tech Stack Recommender")
print("=" * 45)
print("\nEnter at least 3 skills separated by commas.")
print("Example: Python, Machine Learning, SQL\n")

user_input = input("Your skills: ")
user_skills = [s.strip() for s in user_input.split(",")]

if len(user_skills) < 3:
    print("\n[ERROR] Please provide at least 3 skills.")
    exit()

user_profile = ", ".join(user_skills)

# load data + vector mapping
df = pd.read_csv("raw_skills.csv")

all_documents = list(df["skills"]) + [user_profile]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_documents)

job_vectors = tfidf_matrix[:-1]
user_vector = tfidf_matrix[-1]

# scoring
scores = cosine_similarity(user_vector, job_vectors).flatten()
df["similarity_score"] = scores

# sort+filter
top_n = 3
recommendations = df.sort_values(
    "similarity_score", ascending=False
).head(top_n).reset_index(drop=True)

# output
print(f"\n{'=' * 45}")
print("   Top {n} Career Paths For You".format(n=top_n))
print(f"{'=' * 45}\n")

for i, row in recommendations.iterrows():
    score_pct = round(row["similarity_score"] * 100, 1)
    print(f"  {i+1}  {row['job_role']}")
    print(f"       Match  : {score_pct}%")
    print(f"       Skills : {row['skills']}")
    print()

print("=" * 45)
print("  Powered by DecodeLabs | Batch 2026")
print("=" * 45)
