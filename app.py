import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
from openai import OpenAI

# Set page config
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Font size + GPT button styling
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-size: 18px !important;
}
.stDataFrame th, .stDataFrame td {
    font-size: 16px !important;
}
h1, h2, h3, h4 {
    font-size: 24px !important;
}
.stMarkdown {
    font-size: 18px !important;
}
div.stButton > button:first-child {
    font-size: 18px;
    height: 3em;
    width: 100%;
    background-color: #f63366;
    color: white;
    border-radius: 8px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/fluency/48/heart-with-pulse.png", width=48)
st.sidebar.title("Heart Attack Risk Predictor")
st.sidebar.markdown("""
### 📝 Instructions
- **Upload** your data file (.csv or .parquet)
- **Select** a baseline model (optional)
- **View** predictions and SHAP explanations
- **Compare** model performance
- **Or** use Manual Entry tab to input key features
""")

baseline_model = st.sidebar.selectbox(
    "Baseline model for comparison (optional):",
    ["None", "Logistic Regression"],
    key="baseline_model"
)

# --- Helper Mappings Dropdown ---
income_mapping = {
    1: "< $10,000",
    2: "$10,000 - $15,000",
    3: "$15,000 - $20,000",
    4: "$20,000 - $25,000",
    5: "$25,000 - $35,000",
    6: "$35,000 - $50,000",
    7: "$50,000 - $75,000",
    8: ">= $75,000"
}
age_group_mapping = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80+"
}

mapping_choice = st.sidebar.selectbox(
    "View Category Mappings:",
    ["Income", "Age Group"],
    key="mapping_choice",
    help="See how dropdown values map to underlying codes"
)
if mapping_choice == "Income":
    st.sidebar.table(
        pd.DataFrame(list(income_mapping.items()), columns=["Code", "Income Range"]))
elif mapping_choice == "Age Group":
    st.sidebar.table(
        pd.DataFrame(list(age_group_mapping.items()), columns=["Code", "Age Range"]))

# --- MAIN ---
st.title("Heart Attack Risk Prediction Demo")
st.write("Choose between uploading a file or manually entering key features.")

uploaded_file = st.file_uploader(
    "Upload your data file (CSV or Parquet)",
    type=["csv", "parquet"],
    help="File must have the same columns as the training pipeline."
)

@st.cache_resource
def load_models():
    try:
        return joblib.load('models/xgb_top25_shap.joblib')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

xgb_pipeline = load_models()


def clean_feature_name(name):
    return name.split("__", 1)[1] if "__" in name else name


def feature_engineering(df):
    df = df.copy()
    combo_flags = ["prev_chd_or_mi", "smoked_100_cigs", "high_bp_flag", "stroke", "taking_bp_meds"]
    if all(col in df.columns for col in combo_flags):
        df["high_risk_combo_sum"] = df[combo_flags].sum(axis=1)
        df["high_risk_cluster_2plus"] = (df["high_risk_combo_sum"] >= 2).astype(int)
    if "bmi_log_z" in df.columns and "physical_unhealthy_days_log_z" in df.columns:
        df["bmi_x_phys_unhealthy"] = df["bmi_log_z"] * df["physical_unhealthy_days_log_z"]
    if "smoked_100_cigs" in df.columns and "age_group" in df.columns:
        df["smoker_age_risk"] = df["smoked_100_cigs"] * df["age_group"]
    chronic = ["asthma", "stroke", "high_bp_flag", "skin_cancer", "other_cancer"]
    df["chronic_burden"] = df[[c for c in chronic if c in df.columns]].sum(axis=1)
    if "has_primary_doctor" in df.columns and "chol_check_recent" in df.columns:
        df["preventive_neglect"] = ((df["has_primary_doctor"] == 0) & (df["chol_check_recent"] == 0)).astype(int)
    if "bmi" in df.columns:
        df["bmi_risk_category"] = pd.cut(
            df["bmi"], bins=[0,18.5,24.9,29.9,100],
            labels=["underweight","normal","overweight","obese"]
        )
        df["bmi_risk_code"] = df["bmi_risk_category"].cat.codes.replace(-1, pd.NA)
    return df


def build_shap_prompt(top_features):
    intro = "Given these feature contributions to heart attack risk prediction, explain the outcome:\n"
    lines = [f"- {row['Feature']}: SHAP value = {row['SHAP Value']:.3f}" for _, row in top_features.iterrows()]
    return intro + "\n" + "\n".join(lines)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Preview", "🔮 Predictions", "🧬 SHAP Analysis",
    "🔍 Individual Record Explorer", "✍️ Manual Entry"
])

# File Upload Flow
if uploaded_file and xgb_pipeline:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_parquet(uploaded_file)
    df = feature_engineering(df)
    feature_names = xgb_pipeline.named_steps['pre'].get_feature_names_out()
    pre_data = xgb_pipeline.named_steps['pre'].transform(df)
    preds = xgb_pipeline.named_steps['clf'].predict_proba(pre_data)[:,1]
    classes = (preds >= 0.5).astype(int)
    results_df = pd.DataFrame({'Predicted Class': classes, 'Risk Probability': preds})

    with tab1:
        st.dataframe(df.head(), use_container_width=True)
    with tab2:
        st.dataframe(results_df, use_container_width=True)
    with tab3:
        explainer = shap.TreeExplainer(xgb_pipeline.named_steps['clf'])
        shap_vals = explainer.shap_values(pre_data)
        idx = [i for i,f in enumerate(feature_names) if 'prev_chd_or_mi' not in f]
        feats = [feature_names[i] for i in idx]
        vals = shap_vals[:,idx]
        data = pre_data[:,idx]
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("**Feature Importance (Summary):**")
            fig,ax = plt.subplots(figsize=(7,4))
            shap.summary_plot(vals,data,feature_names=feats,show=False)
            st.pyplot(fig)
            plt.close()
        with col2:
            st.markdown("**Waterfall (First Row):**")
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value, vals[0], feature_names=feats, max_display=10, show=False
            )
            st.pyplot(plt.gcf())
            plt.clf()

    with tab4:
        explorer = df.copy()
        explorer['Risk Probability'] = preds
        explorer['Predicted Class'] = classes
        sel = st.selectbox(
            "Select a record:", range(len(explorer)), key="record_select",
            format_func=lambda i: f"Record {i+1} (Risk: {explorer.iloc[i]['Risk Probability']:.2%})"
        )
        row = explorer.iloc[sel]
        # Display raw feature values
        st.markdown("#### Input Feature Values for Selected Record")
        raw_vals = {col: row.get(col) for col in ['income','smoker_age_risk','age_group','bmi','physical_unhealthy_days_log_z','high_bp_flag','stroke']}
        raw = {k:v for k,v in raw_vals.items() if pd.notnull(v)}
        st.json(raw)
        col1,col2 = st.columns(2)
        with col1:
            st.metric("Risk Probability",f"{row['Risk Probability']:.2%}")
            st.metric("Predicted Class","High Risk" if row['Predicted Class']==1 else"Low Risk")
        with col2:
            prob=row['Risk Probability']
            clr = "green" if prob<0.3 else "orange" if prob<0.7 else "red"
            st.progress(float(prob),text=f"{prob:.2%}")
            st.markdown(f"<style>div.stProgress > div > div > div > div{{background-color:{clr};}}</style>",unsafe_allow_html=True)

        st.markdown("### Top SHAP Contributors")
        shap_df = pd.DataFrame({'Feature':feats,'SHAP':vals[sel]})
        top5 = shap_df.reindex(shap_df['SHAP'].abs().nlargest(5).index)
        fig,ax=plt.subplots(figsize=(10,4))
        colors=['red' if v>0 else 'green' for v in top5['SHAP']]
        ax.barh(top5['Feature'],top5['SHAP'],color=colors)
        for bar in ax.patches:
            w=bar.get_width()
            ax.text(w,bar.get_y()+bar.get_height()/2,f"{w:.3f}",va='center',ha='left' if w>0 else 'right')
        plt.title('Top 5 Feature Contributions (Signed)')
        plt.xlabel('SHAP Value')
        st.pyplot(fig)

        st.markdown("### Key Factors Contributing to Prediction")
        base=explainer.expected_value;pred_sum=base+top5['SHAP'].sum()
        expl_text=[f"• {f}: {'increases' if v>0 else 'decreases'} risk by {abs(v):.3f}" for f,v in zip(top5['Feature'],top5['SHAP'])]
        st.markdown("\n".join(expl_text))
        st.markdown(f"**Overall**: These factors contribute to a {'**high**' if pred_sum>0.5 else '**low**'} risk prediction (prob: {row['Risk Probability']:.2%})")

        if st.button("🧠 Explain in Plain English", key="gpt_explain"):
            prompt = build_shap_prompt(top5.rename(columns={'SHAP':'SHAP Value'}))
            try:
                client=OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
                resp=client.chat.completions.create(model="gpt-4o",messages=[{'role':'user','content':prompt}],temperature=0.4,max_tokens=250)
                st.markdown("### 📝 AI Explanation")
                st.write(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"AI error: {e}")

# Manual Entry Flow
with tab5:
    st.markdown("### Enter Key Features Manually")
    manual_defaults = {}
    if xgb_pipeline:
        raw_cols = list(xgb_pipeline.named_steps['pre'].feature_names_in_)
        # initialize with zeros
        manual_defaults = {col: 0.0 for col in raw_cols}
    with st.form("manual_form"):
        age = st.number_input("Age Group (1–13)", min_value=1, max_value=13, value=5, key="manual_age")
        sex = st.selectbox("Sex", ["Male", "Female"], key="manual_sex")
        smoked = st.selectbox("Smoked 100+ Cigarettes?", [0, 1], key="manual_smoked")
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, key="manual_bmi")
        phys = st.number_input(
            "Physical Unhealthy Days (log-z)", min_value=0.0, max_value=5.0, value=0.5, key="manual_phys"
        )
        income = st.selectbox("Income Category (1–8)", list(range(1, 9)), key="manual_income")
        submitted = st.form_submit_button("Predict from Manual Input")

    if submitted:
        # Create base row and set manual inputs
        manual_row = pd.DataFrame([manual_defaults])
        manual_row['age_group'] = age
        manual_row['smoked_100_cigs'] = smoked
        manual_row['bmi'] = bmi
        manual_row['physical_unhealthy_days_log_z'] = phys
        manual_row['income'] = income
        manual_row['sex_Female'] = 1 if sex == 'Female' else 0

        # Feature engineering
        fe_tmp = feature_engineering(manual_row)
        # Build final input frame with expected columns
        input_cols = xgb_pipeline.named_steps['pre'].feature_names_in_
        manual_fe = pd.DataFrame([manual_defaults]).astype(float)
        for col in fe_tmp.columns:
            if col in manual_fe.columns:
                manual_fe[col] = fe_tmp[col].astype(float)

        # Preprocess and predict
        try:
            proc = xgb_pipeline.named_steps['pre'].transform(manual_fe)
            pred = xgb_pipeline.named_steps['clf'].predict_proba(proc)[0, 1]
            cls = int(pred >= 0.5)
            st.metric("Risk Probability", f"{pred:.2%}")
            st.metric("Predicted Class", "High Risk" if cls == 1 else "Low Risk")

            # SHAP explanation
            explainer = shap.TreeExplainer(xgb_pipeline.named_steps['clf'])
            sv = explainer.shap_values(proc)[0]
            fnames = xgb_pipeline.named_steps['pre'].get_feature_names_out()
            df_shap = pd.DataFrame({'Feature': fnames, 'SHAP': sv})
            top5 = df_shap.reindex(df_shap['SHAP'].abs().nlargest(5).index)

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['red' if v > 0 else 'green' for v in top5['SHAP']]
            ax.barh(top5['Feature'], top5['SHAP'], color=colors)
            plt.title('Top 5 SHAP Contributors (Manual)')
            for bar in ax.patches:
                w = bar.get_width()
                ax.text(
                    w,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}",
                    va='center',
                    ha='left' if w > 0 else 'right'
                )
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Manual prediction error: {e}")

# Baseline Comparison
if uploaded_file and xgb_pipeline and baseline_model!="None":
    df_shared=xgb_pipeline.named_steps['pre'].transform(df)
    base=LogisticRegression(random_state=42).fit(df_shared,classes)
    bp=base.predict_proba(df_shared)[:,1]
    comp=pd.DataFrame({'XGB':preds,'Baseline':bp})
    st.subheader("⚖️ Model Comparison")
    st.dataframe(comp)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=comp['XGB'],y=comp['Baseline'],mode='markers'))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(dash='dash')))
    st.plotly_chart(fig)
else:
    if not uploaded_file:
        st.info("Please upload a data file to get started.")
    elif not xgb_pipeline:
        st.error("Model not loaded.")