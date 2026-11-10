# Mall Customer Segmentation — training and evaluation
# Expected CSV columns: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "Mall_Customers.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(CSV)
print("\nFirst five records:\n", df.head())
print("\nDimensions:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nNumerical columns:", df.select_dtypes(include="number").columns.tolist())
print("\nCategorical columns:", df.select_dtypes(exclude="number").columns.tolist())

df = df.drop_duplicates().copy()
numeric = df.select_dtypes(include="number").columns
for col in numeric:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(exclude="number").columns:
    mode = df[col].mode()
    if len(mode):
        df[col] = df[col].fillna(mode.iloc[0])

features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].copy()
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Four EDA visualizations
for col, filename, title in [
    ("Annual Income (k$)", "income_distribution.png", "Annual Income Distribution"),
    ("Spending Score (1-100)", "spending_distribution.png", "Spending Score Distribution"),
    ("Age", "age_distribution.png", "Age Distribution"),
]:
    plt.figure()
    df[col].hist(bins=15)
    plt.title(title); plt.xlabel(col); plt.ylabel("Number of customers")
    plt.tight_layout(); plt.savefig(OUT / filename, dpi=160); plt.close()

plt.figure()
plt.scatter(df[features[0]], df[features[1]], alpha=.75)
plt.title("Annual Income vs Spending Score")
plt.xlabel(features[0]); plt.ylabel(features[1])
plt.tight_layout(); plt.savefig(OUT / "income_vs_spending.png", dpi=160); plt.close()

# K search: use silhouette to guide selection; inspect elbow and plot before finalizing.
scores, inertias = [], []
for k in range(2, min(11, len(df))):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    scores.append((k, silhouette_score(Xs, labels)))
    inertias.append((k, km.inertia_))
print("\nK-Means silhouette scores:", scores)
print("K-Means inertia:", inertias)

plt.figure()
plt.plot([x[0] for x in inertias], [x[1] for x in inertias], marker="o")
plt.title("K-Means Elbow Curve"); plt.xlabel("Number of clusters (K)"); plt.ylabel("Inertia")
plt.tight_layout(); plt.savefig(OUT / "elbow_curve.png", dpi=160); plt.close()

# Select best silhouette among tested K values; inspect resulting clusters for interpretability.
best_k = max(scores, key=lambda item: item[1])[0]
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(Xs)
print(f"\nSelected K={best_k}; silhouette={silhouette_score(Xs, labels):.4f}; inertia={kmeans.inertia_:.4f}")

agg = AgglomerativeClustering(n_clusters=best_k)
agg_labels = agg.fit_predict(Xs)
agg_score = silhouette_score(Xs, agg_labels)
print(f"Agglomerative silhouette at K={best_k}: {agg_score:.4f}")

plt.figure()
plt.plot([x[0] for x in scores], [x[1] for x in scores], marker="o", label="K-Means")
plt.axhline(agg_score, linestyle="--", label="Agglomerative at selected K")
plt.title("Silhouette Score Comparison"); plt.xlabel("K"); plt.ylabel("Silhouette score")
plt.legend(); plt.tight_layout(); plt.savefig(OUT / "silhouette_comparison.png", dpi=160); plt.close()

df["Cluster"] = labels
plt.figure()
for cluster in sorted(df["Cluster"].unique()):
    part = df[df["Cluster"] == cluster]
    plt.scatter(part[features[0]], part[features[1]], label=f"Cluster {cluster}", alpha=.8)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=180, label="Centroids")
plt.title("Customer Segments"); plt.xlabel(features[0]); plt.ylabel(features[1])
plt.legend(); plt.tight_layout(); plt.savefig(OUT / "customer_clusters.png", dpi=160); plt.close()

print("\nCluster profiles (mean feature values):\n", df.groupby("Cluster")[features].mean().round(2))
joblib.dump({"model": kmeans, "scaler": scaler, "features": features, "k": best_k}, ROOT / "customer_segment_model.joblib")
df.to_csv(OUT / "customers_with_clusters.csv", index=False)
print(f"\nSaved charts and clustered data in {OUT}; saved model in project folder.")
