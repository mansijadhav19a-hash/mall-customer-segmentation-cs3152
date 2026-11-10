import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Mall Customer Intelligence", page_icon="🛍️", layout="wide")
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "customer_segment_model.joblib"
CSV_PATH = ROOT / "Mall_Customers.csv"
OUT = ROOT / "outputs"

st.markdown("""
<style>
.block-container {padding-top:1.5rem; max-width:1250px;}
.hero {padding:28px; border-radius:16px; color:white;
background:linear-gradient(115deg,#14213d,#264653,#287271); margin-bottom:20px;}
.hero h1 {color:white; margin:0 0 8px 0;}
.hero p {color:#e8f1f2; margin:0;}
div[data-testid="stMetric"] {background:#f3f6fa; padding:14px; border-radius:12px; border:1px solid #e1e8f0;}
</style>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train_model.py` in the project folder first.")
    st.stop()

saved = joblib.load(MODEL_PATH)
model, scaler, features, k = saved["model"], saved["scaler"], saved["features"], saved["k"]
df = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else None
clustered_path = OUT / "customers_with_clusters.csv"
clustered = pd.read_csv(clustered_path) if clustered_path.exists() else None
centres = scaler.inverse_transform(model.cluster_centers_)
profiles = pd.DataFrame(centres, columns=features)
income_mid = float(profiles[features[0]].median())
spending_mid = float(profiles[features[1]].median())

def describe(income, spending):
    if income >= income_mid and spending >= spending_mid:
        return "High Income · High Spending"
    if income >= income_mid and spending < spending_mid:
        return "High Income · Low Spending"
    if income < income_mid and spending >= spending_mid:
        return "Lower Income · High Spending"
    return "Lower Income · Low Spending"

with st.sidebar:
    st.markdown("## 🛍️ Mall Intelligence")
    st.caption("Machine Learning · Unsupervised Learning")
    st.divider()
    st.markdown("**Project details**")
    st.write("**Student:** Mansi Sanjay Jadhav")
    st.write("**Class:** TY A CSE")
    st.write("**Roll No.:** CS3152")
    st.write("**Algorithm:** K-Means")
    st.write("**Dataset:** Mall Customers")
    st.divider()
    st.caption("Cluster numbers are labels, not rankings.")

st.markdown("""
<div class="hero">
<h1>🛍️ Mall Customer Segmentation</h1>
<p>Customer intelligence dashboard powered by K-Means clustering</p>
<p style="margin-top:12px;">Explore customer groups, understand spending patterns, and assign a new customer to a learned segment.</p>
</div>
""", unsafe_allow_html=True)

a,b,c,d = st.columns(4)
a.metric("Customer records", f"{len(df):,}" if df is not None else "Dataset loaded")
b.metric("Discovered segments", str(k))
c.metric("ML algorithm", "K-Means")
d.metric("Input features", "2")

overview, predict_tab, insights, methodology = st.tabs([
    "📊 Project Overview", "🔎 Predict Customer Segment", "📈 Data Insights", "📘 Methodology"
])

with overview:
    st.subheader("Project overview")
    st.write("This system uses **unsupervised machine learning** to group mall customers by annual income and spending score. It discovers patterns without pre-labelled segment categories.")
    left,right = st.columns(2)
    with left:
        st.markdown("#### System workflow")
        st.markdown("1. Load customer dataset.\n2. Inspect and clean the data.\n3. Scale selected numerical features.\n4. Apply K-Means clustering.\n5. Evaluate and compare clusters.\n6. Save the trained model.\n7. Assign new customers to a segment.")
    with right:
        st.markdown("#### Model configuration")
        st.metric("Selected number of clusters (K)", k)
        st.write("**Features:** Annual Income (k$), Spending Score (1–100)")
        st.write("**Learning type:** Unsupervised clustering")
    st.subheader("Customer segment profiles")
    table = profiles.copy()
    table.insert(0, "Cluster", range(k))
    table["Segment interpretation"] = [describe(r[features[0]],r[features[1]]) for _,r in table.iterrows()]
    st.dataframe(table[["Cluster","Segment interpretation"]+features].round(2), use_container_width=True, hide_index=True)
    st.caption("Profile values are cluster-centre averages in the original feature units.")

with predict_tab:
    st.subheader("Assign a customer to a segment")
    st.write("Enter the customer's annual income and spending score. The saved scaler and trained K-Means model will process the input.")
    col1,col2 = st.columns([1,1.1], gap="large")
    with col1:
        with st.container(border=True):
            income = st.number_input("Annual income (k$)", min_value=0.0, max_value=500.0, value=60.0, step=1.0)
            spending = st.slider("Spending score (1–100)", 1, 100, 50)
            go = st.button("✨ Analyse customer", type="primary", use_container_width=True)
    with col2:
        with st.container(border=True):
            st.markdown("#### Prediction result")
            if go:
                scaled = scaler.transform(np.array([[income,spending]],dtype=float))
                cluster = int(model.predict(scaled)[0])
                centre = profiles.iloc[cluster]
                st.success(f"Assigned to Cluster {cluster}")
                st.markdown(f"### {describe(centre[features[0]],centre[features[1]])}")
                x,y = st.columns(2)
                x.metric("Annual income",f"{income:.1f} k$")
                y.metric("Spending score",f"{spending}/100")
                st.write("**Cluster centre**")
                st.write(f"Annual income: {centre[features[0]]:.2f} k$")
                st.write(f"Spending score: {centre[features[1]]:.2f}")
            else:
                st.info("Enter customer details and click Analyse customer to view the result.")

with insights:
    st.subheader("Exploratory data analysis")
    if df is None:
        st.warning("Mall_Customers.csv was not found in the project folder.")
    else:
        x,y = st.columns(2)
        for col, title, slot in [
            ("Annual Income (k$)","Annual income distribution",x),
            ("Spending Score (1-100)","Spending score distribution",y)
        ]:
            with slot:
                st.markdown("#### "+title)
                fig,ax = plt.subplots()
                df[col].hist(bins=15,ax=ax)
                ax.set_xlabel(col); ax.set_ylabel("Customer count")
                st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.markdown("#### Customer groups")
        if clustered is not None and "Cluster" in clustered.columns:
            fig,ax = plt.subplots(figsize=(9,5))
            for cl in sorted(clustered["Cluster"].unique()):
                part = clustered[clustered["Cluster"]==cl]
                ax.scatter(part[features[0]],part[features[1]],label=f"Cluster {cl}",alpha=.75)
            ax.scatter(centres[:,0],centres[:,1],marker="X",s=180,label="Cluster centres")
            ax.set_xlabel(features[0]); ax.set_ylabel(features[1]); ax.legend(ncol=2); ax.grid(alpha=.2)
            st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.markdown("#### Dataset preview")
        st.dataframe(df.head(10),use_container_width=True,hide_index=True)
    st.markdown("#### Model evaluation from the completed training run")
    st.metric("K-Means silhouette score","0.5547")
    st.metric("Agglomerative silhouette score","0.5538")
    st.caption("These are the scores shown in the current completed run. Retraining may change them.")

with methodology:
    st.subheader("Machine learning methodology")
    st.markdown("""
**1. Problem type — Unsupervised learning**  
There are no pre-labelled customer segment targets. The algorithm discovers groups from feature similarity.

**2. Feature selection**  
Annual Income (k$) and Spending Score (1–100) represent purchasing capacity and spending behaviour.

**3. Preprocessing**  
Duplicate rows are removed, missing numerical values are filled using the median, and selected features are standardized. Scaling is important because K-Means uses distance calculations.

**4. K-Means clustering**  
K-Means assigns each customer to the nearest centroid and updates centroids until the groups stabilize.

**5. Evaluation**  
The silhouette score measures cluster separation and cohesion. Inertia measures within-cluster squared distances and is interpreted with the elbow curve.

**6. New-customer prediction**  
The app applies the same fitted scaler used during training and then predicts the nearest learned cluster.
""")
    st.warning("Limitations: results depend on the dataset and selected features. Cluster IDs are arbitrary labels and do not explain why a person spends a certain amount.")
    st.markdown("**Future scope:** test additional suitable customer features, use updated customer data, and validate whether the segments are useful for real mall decisions.")

st.divider()
st.caption("Individual ML Project · Mansi Sanjay Jadhav · TY A CSE · Roll No. CS3152")
