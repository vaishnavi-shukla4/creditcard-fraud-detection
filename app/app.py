import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💳 Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("../models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("../models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("../models/features.pkl", "rb") as f:
        features = pickle.load
        
        (f)
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("💳 Fraud Detector")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏠 Overview", "🔍 Predict Transaction", "📊 Model Insights"])
st.sidebar.markdown("---")
st.sidebar.info("**Dataset:** Kaggle Credit Card Fraud  \n**Model:** XGBoost  \n**Best AUC:** ~0.98")

# ── Page 1: Overview ──────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("💳 Credit Card Fraud Detection")
    st.markdown("### Detecting fraudulent transactions using Machine Learning")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", "284,807")
    col2.metric("Fraud Cases", "492", delta="0.17%")
    col3.metric("Model", "XGBoost")
    col4.metric("ROC-AUC", "~0.98", delta="+0.08 vs baseline")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🔴 The Imbalance Problem")
        st.markdown("""
        The dataset is **severely imbalanced** — only **0.17%** of transactions are fraudulent.
        
        This means:
        - A model that predicts *everything as legit* achieves **99.83% accuracy**
        - We need **Precision-Recall AUC**, not just accuracy
        - We use **SMOTE** to synthetically oversample the minority class
        """)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(['Legitimate', 'Fraud'], [284315, 492], color=['steelblue', 'crimson'])
        ax.set_title('Class Distribution')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    with col_b:
        st.subheader("📈 Model Comparison")
        comparison_data = {
            'Model': ['LR Baseline', 'LR + SMOTE', 'Random Forest', 'XGBoost'],
            'ROC-AUC': [0.970, 0.974, 0.979, 0.983],
            'Avg Precision': [0.70, 0.72, 0.85, 0.87],
            'Recall (Fraud)': [0.60, 0.76, 0.82, 0.84]
        }
        df_comp = pd.DataFrame(comparison_data).set_index('Model')
        st.dataframe(df_comp.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

        st.markdown("""
        **Key takeaway:** XGBoost with `scale_pos_weight` achieves the best balance
        of precision and recall for fraud detection.
        """)

    st.markdown("---")
    st.subheader("💡 Key Insights from EDA")
    i1, i2, i3 = st.columns(3)
    i1.success("🕐 Fraud peaks during **off-peak hours** (late night / early morning)")
    i2.warning("💰 Fraud transactions are **smaller in amount** on average to avoid detection")
    i3.error("📉 Features **V14, V17, V12** are the strongest fraud signals")


# ── Page 2: Predict ───────────────────────────────────────────────────────────
elif page == "🔍 Predict Transaction":
    st.title("🔍 Predict a Transaction")
    st.markdown("Enter transaction details below to check if it's fraudulent.")

    tab1, tab2 = st.tabs(["✏️ Manual Input", "📁 Upload CSV"])

    with tab1:
        st.subheader("Transaction Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=30000.0, value=150.0, step=10.0)
            time_val = st.number_input("Time (seconds since first transaction)", min_value=0, max_value=172800, value=50000)

        with col2:
            st.markdown("**PCA Features (V1–V14)**")
            v_vals = {}
            for i in range(1, 15):
                v_vals[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

        with col3:
            st.markdown("**PCA Features (V15–V28)**")
            for i in range(15, 29):
                v_vals[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

        if st.button("🚨 Analyze Transaction", type="primary", use_container_width=True):
            # Build input
            amount_scaled = scaler.transform([[amount]])[0][0]
            time_scaled = (time_val - 94813) / 47488  # approximate normalization

            input_dict = {**v_vals, 'Amount_Scaled': amount_scaled, 'Time_Scaled': time_scaled}
            input_df = pd.DataFrame([input_dict])[feature_names]

            prob = model.predict_proba(input_df)[0][1]
            pred = int(prob >= 0.5)

            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if pred == 1:
                    st.error(f"### 🚨 FRAUD DETECTED")
                    st.metric("Fraud Probability", f"{prob*100:.1f}%")
                else:
                    st.success(f"### ✅ LEGITIMATE")
                    st.metric("Fraud Probability", f"{prob*100:.1f}%")

                # Gauge-style bar
                fig, ax = plt.subplots(figsize=(4, 0.6))
                color = 'crimson' if prob > 0.5 else 'steelblue'
                ax.barh(['Risk'], [prob], color=color, height=0.5)
                ax.barh(['Risk'], [1 - prob], left=[prob], color='#e0e0e0', height=0.5)
                ax.set_xlim(0, 1)
                ax.axvline(0.5, color='black', linestyle='--', linewidth=1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                ax.set_title('Fraud Risk Score')
                st.pyplot(fig)

            with res_col2:
                st.markdown("**SHAP Explanation — Why this prediction?**")
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(input_df)
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    shap.waterfall_plot(
                        shap.Explanation(values=shap_vals[0],
                                         base_values=explainer.expected_value,
                                         data=input_df.values[0],
                                         feature_names=feature_names),
                        show=False
                    )
                    st.pyplot(plt.gcf())
                except Exception:
                    top_features = pd.Series(
                        dict(zip(feature_names, np.abs(model.feature_importances_)))
                    ).nlargest(10)
                    fig2, ax2 = plt.subplots(figsize=(7, 4))
                    top_features.sort_values().plot(kind='barh', ax=ax2, color='steelblue')
                    ax2.set_title('Top Feature Importances')
                    st.pyplot(fig2)

    with tab2:
        st.subheader("Batch Prediction via CSV Upload")
        st.info("Upload a CSV with the same format as creditcard.csv (without the Class column)")
        uploaded = st.file_uploader("Upload CSV", type=['csv'])

        if uploaded:
            batch_df = pd.read_csv(uploaded)
            st.write(f"Loaded {len(batch_df):,} rows")

            # Preprocess
            if 'Amount' in batch_df.columns:
                batch_df['Amount_Scaled'] = scaler.transform(batch_df[['Amount']])
            if 'Time' in batch_df.columns:
                batch_df['Time_Scaled'] = (batch_df['Time'] - 94813) / 47488
                batch_df.drop(columns=['Time', 'Amount'], inplace=True, errors='ignore')
            if 'Class' in batch_df.columns:
                batch_df.drop(columns=['Class'], inplace=True)

            try:
                input_batch = batch_df[feature_names]
                probs = model.predict_proba(input_batch)[:, 1]
                preds = (probs >= 0.5).astype(int)

                batch_df['Fraud_Probability'] = probs
                batch_df['Prediction'] = preds
                batch_df['Result'] = batch_df['Prediction'].map({0: '✅ Legitimate', 1: '🚨 Fraud'})

                flagged = preds.sum()
                st.error(f"🚨 {flagged} fraudulent transactions detected out of {len(preds):,}")

                st.dataframe(
                    batch_df[['Fraud_Probability', 'Result']].style.applymap(
                        lambda x: 'background-color: #ffcccc' if x == '🚨 Fraud' else '',
                        subset=['Result']
                    ),
                    use_container_width=True
                )

                csv_out = batch_df.to_csv(index=False)
                st.download_button("⬇️ Download Results", csv_out, "fraud_predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Column mismatch: {e}")


# ── Page 3: Model Insights ────────────────────────────────────────────────────
elif page == "📊 Model Insights":
    st.title("📊 Model Insights")

    st.subheader("Feature Importance (XGBoost)")
    importances = pd.Series(
        dict(zip(feature_names, model.feature_importances_))
    ).nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    importances.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title('Top 15 Features by Importance', fontweight='bold')
    ax.set_xlabel('Importance Score')
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📚 Understanding the Model")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Why XGBoost?**
        - Handles imbalanced data via `scale_pos_weight`
        - Robust to outliers in PCA features
        - Built-in feature importance
        - Fastest to train among ensemble methods

        **Metrics that matter:**
        - ✅ **Recall** — catching actual fraud (minimize false negatives)
        - ✅ **Precision** — avoiding false alarms
        - ✅ **PR-AUC** — best single metric for imbalanced problems
        - ❌ **Accuracy** — misleading here (99.83% trivially)
        """)
    with c2:
        st.markdown("""
        **Handling Imbalance:**
        - **SMOTE:** Synthetically creates minority class samples
        - **scale_pos_weight:** Tells XGBoost to penalize fraud misses more
        - **Threshold tuning:** Move from 0.5 to optimize precision/recall tradeoff

        **Business Impact:**
        - Each missed fraud = financial loss + customer trust damage
        - Each false alarm = customer inconvenience
        - Optimal threshold depends on the cost ratio of these two errors
        """)