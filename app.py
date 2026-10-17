import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    cluster_names = joblib.load("cluster_names.pkl")
    return model, scaler, cluster_names

try:
    kmeans_model, scaler, cluster_names = load_models()
except Exception as e:
    st.error("Could not load the model files. Keep kmeans_model.pkl, scaler.pkl, and cluster_names.pkl in the same folder as app.py.")
    st.exception(e)
    st.stop()

# -------------------- HELPERS --------------------
def get_cluster_name(cluster_id):
    try:
        if isinstance(cluster_names, dict):
            return cluster_names.get(cluster_id, cluster_names.get(str(cluster_id), f"Cluster {cluster_id}"))
        return cluster_names[cluster_id]
    except (IndexError, KeyError, TypeError):
        return f"Cluster {cluster_id}"

def predict_cluster(age, income, spending):
    # Feature order must match the order used to train the scaler.
    data = pd.DataFrame(
        [[age, income, spending]],
        columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    )
    scaled = scaler.transform(data)
    return int(kmeans_model.predict(scaled)[0])

# Optional sample data makes the dashboard visualizations useful before any CSV is uploaded.
def make_demo_data():
    rng = np.random.default_rng(42)
    centers = [(25, 25, 80), (45, 85, 20), (32, 80, 80), (50, 45, 50)]
    rows = []
    for age_c, income_c, spending_c in centers:
        for _ in range(35):
            age = int(np.clip(rng.normal(age_c, 7), 18, 75))
            income = float(np.clip(rng.normal(income_c, 12), 10, 150))
            spending = int(np.clip(rng.normal(spending_c, 12), 1, 100))
            rows.append([age, income, spending])
    return pd.DataFrame(rows, columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"])

def assign_clusters(df):
    required = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    clean = df[required].dropna().copy()
    clean["Cluster"] = kmeans_model.predict(scaler.transform(clean[required]))
    clean["Segment"] = clean["Cluster"].apply(lambda x: str(get_cluster_name(int(x))))
    return clean

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("🛍️ Segmentation AI")
    st.caption("Customer analytics workspace")
    page = st.radio(
        "NAVIGATION",
        ["📊 Dashboard", "🎯 Predict Customer", "📁 Upload Customer Data", "ℹ️ About Project"]
    )
    st.divider()
    st.markdown("### Model Information")
    st.write("**Algorithm:** K-Means")
    st.write("**Clusters:** 4")
    st.write("**Features:** 3")
    st.caption("Age • Annual Income • Spending Score")
    st.divider()
    st.caption("Customer Segmentation Project")

# -------------------- SESSION DATA --------------------
if "customers_df" not in st.session_state:
    st.session_state.customers_df = make_demo_data()
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# -------------------- DASHBOARD --------------------
if page == "📊 Dashboard":
    st.title("📊 Customer Segmentation Dashboard")
    st.markdown("Understand your customers, explore segment patterns, and identify spending behavior.")
    st.divider()

    df = st.session_state.customers_df
    clustered = assign_clusters(df)
    cluster_count = clustered["Cluster"].nunique()
    total_customers = len(clustered)
    avg_age = clustered["Age"].mean() if total_customers else 0
    avg_income = clustered["Annual Income (k$)"].mean() if total_customers else 0
    avg_spending = clustered["Spending Score (1-100)"].mean() if total_customers else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers Analyzed", f"{total_customers:,}")
    c2.metric("Clusters Found", f"{cluster_count}/4")
    c3.metric("Average Income", f"${avg_income:.1f}k")
    c4.metric("Average Spending Score", f"{avg_spending:.1f}/100")

    st.markdown("### Customer Insights")
    left, right = st.columns(2)

    with left:
        st.markdown("#### Income vs. Spending")
        fig, ax = plt.subplots(figsize=(7, 4))
        for cid in sorted(clustered["Cluster"].unique()):
            subset = clustered[clustered["Cluster"] == cid]
            ax.scatter(
                subset["Annual Income (k$)"],
                subset["Spending Score (1-100)"],
                label=str(get_cluster_name(int(cid))),
                alpha=0.75
            )
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1-100)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("#### Customers by Cluster")
        counts = clustered["Segment"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_ylabel("Number of Customers")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    left2, right2 = st.columns(2)
    with left2:
        st.markdown("#### Average Spending Score by Segment")
        profile = clustered.groupby("Segment", sort=False)["Spending Score (1-100)"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(profile.index.astype(str), profile.values)
        ax.set_xlabel("Average Spending Score")
        ax.grid(axis="x", alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right2:
        st.markdown("#### Segment Summary")
        summary = clustered.groupby(["Cluster", "Segment"], as_index=False).agg(
            Customers=("Age", "size"),
            **{
                "Average Age": ("Age", "mean"),
                "Average Income (k$)": ("Annual Income (k$)", "mean"),
                "Average Spending": ("Spending Score (1-100)", "mean")
            }
        )
        for col in ["Average Age", "Average Income (k$)", "Average Spending"]:
            summary[col] = summary[col].round(1)
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.caption("If no CSV is uploaded, the dashboard uses illustrative sample data. Upload your customer dataset for real analytics.")

# -------------------- PREDICT CUSTOMER --------------------
elif page == "🎯 Predict Customer":
    st.title("🎯 Predict Customer Segment")
    st.markdown("Enter a customer's details to predict their segment.")
    with st.container(border=True):
        with st.form("prediction_form"):
            a, b, c = st.columns(3)
            age = a.number_input("Age (years)", min_value=1, max_value=100, value=25)
            income = b.number_input("Annual Income (k$)", min_value=0.0, max_value=500.0, value=50.0, step=1.0)
            spending = c.slider("Spending Score (1–100)", min_value=1, max_value=100, value=50)
            submitted = st.form_submit_button("✨ Predict Segment", use_container_width=True)

    if submitted:
        cluster_id = predict_cluster(age, income, spending)
        segment = get_cluster_name(cluster_id)
        st.session_state.last_prediction = {
            "Age": age,
            "Annual Income (k$)": income,
            "Spending Score (1-100)": spending,
            "Cluster": cluster_id,
            "Segment": str(segment)
        }

    if st.session_state.last_prediction:
        result = st.session_state.last_prediction
        st.divider()
        st.success("Prediction completed successfully!")
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Segment", result["Segment"])
        r2.metric("Cluster Number", result["Cluster"])
        r3.metric("Spending Score", f'{result["Spending Score (1-100)"]}/100')

        st.markdown("### Customer Profile")
        profile_df = pd.DataFrame({
            "Feature": ["Age", "Annual Income (k$)", "Spending Score (1-100)"],
            "Value": [result["Age"], result["Annual Income (k$)"], result["Spending Score (1-100)"]]
        })
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

        # Plot the entered customer against the currently loaded data.
        base = assign_clusters(st.session_state.customers_df)
        fig, ax = plt.subplots(figsize=(8, 4))
        for cid in sorted(base["Cluster"].unique()):
            subset = base[base["Cluster"] == cid]
            ax.scatter(subset["Annual Income (k$)"], subset["Spending Score (1-100)"],
                       label=str(get_cluster_name(int(cid))), alpha=0.55)
        ax.scatter(result["Annual Income (k$)"], result["Spending Score (1-100)"],
                   marker="*", s=260, edgecolors="black", label="Current Customer")
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1-100)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# -------------------- UPLOAD DATA --------------------
elif page == "📁 Upload Customer Data":
    st.title("📁 Upload Customer Dataset")
    st.write("Upload a CSV containing these exact columns to power the dashboard with your data:")
    st.code("Age, Annual Income (k$), Spending Score (1-100)")
    uploaded_file = st.file_uploader("Choose customer CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            required = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
            missing = [col for col in required if col not in uploaded_df.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
                st.info("Rename your CSV columns to match the names shown above.")
            else:
                cleaned = uploaded_df[required].apply(pd.to_numeric, errors="coerce").dropna()
                if cleaned.empty:
                    st.error("No valid customer rows found in the uploaded file.")
                else:
                    st.session_state.customers_df = cleaned.reset_index(drop=True)
                    st.success(f"Loaded {len(cleaned)} customer records.")
                    st.dataframe(cleaned.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")

# -------------------- ABOUT --------------------
else:
    st.title("ℹ️ About Customer Segmentation")
    st.markdown("""
    ### Project Overview
    Customer Segmentation groups customers with similar characteristics using the **K-Means clustering algorithm**.

    ### Features Used
    - **Age:** Customer's age in years
    - **Annual Income (k$):** Annual income in thousands of dollars
    - **Spending Score (1–100):** Customer spending behavior score

    ### How it works
    1. Enter customer details or upload a CSV dataset.
    2. The saved scaler transforms the features.
    3. The trained K-Means model predicts one of four clusters.
    4. The dashboard visualizes the customer segments.

    **Required files:** `kmeans_model.pkl`, `scaler.pkl`, `cluster_names.pkl`
    """)
