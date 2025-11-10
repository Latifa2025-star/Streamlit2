# streamlit_app.py
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="NYC Rodent Inspections — Interactive Dashboard",
    page_icon="🐀",
    layout="wide",
)

# -----------------------------
# Theme / palettes (colorblind-friendly defaults)
# -----------------------------
PAL_SEQ = px.colors.sequential.OrRd
PAL_SEQ_ALT = px.colors.sequential.Sunset
PAL_QUAL = px.colors.qualitative.Set2

# -----------------------------
# Sidebar — controls
# -----------------------------
st.sidebar.title("Controls")
st.sidebar.write("Filter, sample, and model settings")

# Optional row sampling to keep interactive app snappy
sample_n = st.sidebar.number_input(
    "Sample rows (0 = all)", min_value=0, max_value=2_000_000, value=250_000, step=50_000,
    help="Use a sample to keep interactions fast. Set to 0 to load all rows."
)

# Modeling controls
with st.sidebar.expander("Modeling options", expanded=False):
    model_name = st.selectbox("Choose model", ["Random Forest", "Logistic Regression"])
    train_size = st.slider("Train size", 0.5, 0.9, 0.8, 0.05)
    rf_trees = st.number_input("Random Forest trees", 50, 500, 200, 50)
    use_scaler = st.checkbox("Scale numeric features (improves Logistic Regression)", value=True)
    train_samples = st.number_input("Max training rows (for speed)", 50_000, 500_000, 150_000, 25_000)

# -----------------------------
# Data loading & preprocessing
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data(url: str, sample_n: int = 0) -> pd.DataFrame:
    df = pd.read_csv(url)
    if sample_n and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=42)
    return df

@st.cache_data(show_spinner=True)
def clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    # Dates
    df["INSPECTION_DATE"] = pd.to_datetime(df["INSPECTION_DATE"], errors="coerce")
    df["APPROVED_DATE"] = pd.to_datetime(df["APPROVED_DATE"], errors="coerce")
    # restrict to reasonable window (your proposal: 2010–2024)
    df = df[(df["INSPECTION_DATE"] >= "2010-01-01") & (df["INSPECTION_DATE"] <= "2024-12-31")]
    # Year/Month
    df["INSPECTION_YEAR"] = df["INSPECTION_DATE"].dt.year
    df["INSPECTION_MONTH"] = df["INSPECTION_DATE"].dt.month
    # Simple cleaning
    df = df.dropna(subset=["BOROUGH", "RESULT"])
    # Normalize some categories
    df["RESULT"] = df["RESULT"].replace({
        "Bait applied": "Bait applied",
        "Rat Activity": "Rat Activity",
        "Passed": "Passed",
        "Failed for Other R": "Failed for Other R",
        "Monitoring visit": "Monitoring visit"
    })
    return df

DATA_URL = "https://data.cityofnewyork.us/api/views/p937-wjvj/rows.csv?accessType=DOWNLOAD"
with st.spinner("Loading data…"):
    raw = load_data(DATA_URL, sample_n=sample_n)
    df = clean_and_enrich(raw)

st.title("🐀 NYC DOHMH Rodent Inspections — 2010–2024")
st.caption("Interactive EDA, geovisuals, and quick baseline models. Built with Streamlit + Plotly.")

# -----------------------------
# High-level KPIs
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Rows", f"{len(df):,}")
with c2:
    st.metric("Columns", f"{df.shape[1]:,}")
with c3:
    yr_min, yr_max = int(df["INSPECTION_YEAR"].min()), int(df["INSPECTION_YEAR"].max())
    st.metric("Year range", f"{yr_min}–{yr_max}")
with c4:
    st.metric("Boroughs", df["BOROUGH"].nunique())

st.markdown("---")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Time & Seasonality", "Geography", "Model (Preview)"])

# -----------------------------
# TAB 1 — Overview
# -----------------------------
with tab1:
    st.subheader("Outcome Mix & Borough Distribution")

    colA, colB = st.columns([1,1])
    with colA:
        outcome_counts = (
            df["RESULT"].fillna("Unknown")
            .value_counts()
            .rename_axis("RESULT")
            .reset_index(name="count")
        )
        fig_pie = px.pie(
            outcome_counts, names="RESULT", values="count",
            hole=0.45, color="RESULT", color_discrete_sequence=PAL_QUAL,
            title="Inspection Outcomes Mix"
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with colB:
        b_counts = (
            df["BOROUGH"].value_counts().rename_axis("BOROUGH").reset_index(name="count")
        )
        fig_b = px.bar(
            b_counts, x="BOROUGH", y="count",
            color="count", color_continuous_scale=PAL_SEQ_ALT,
            title="Inspections by Borough"
        )
        fig_b.update_layout(xaxis_title="", yaxis_title="Inspections", coloraxis_showscale=False)
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("**Notes**")
    st.write("- ‘Passed’ dominates, while ‘Rat Activity’ is a substantial minority. "
             "Borough volumes are highest in Manhattan/Brooklyn/Bronx.")

# -----------------------------
# TAB 2 — Time & Seasonality
# -----------------------------
with tab2:
    st.subheader("Trends over Time")

    # year trend
    year_counts = (
        df.groupby("INSPECTION_YEAR")
        .size()
        .reset_index(name="count")
        .sort_values("INSPECTION_YEAR")
    )
    fig_year = px.line(
        year_counts, x="INSPECTION_YEAR", y="count",
        markers=True, color_discrete_sequence=[PAL_SEQ[4]],
        title="Inspections per Year (2010–2024)"
    )
    # covid annotation placed inside plot area horizontally
    if 2020 in year_counts["INSPECTION_YEAR"].values:
        y2020 = year_counts.loc[year_counts["INSPECTION_YEAR"]==2020, "count"].values[0]
        fig_year.add_annotation(
            x=2019.4, y=y2020*1.05,
            text="<b>COVID-19 dip</b>",
            showarrow=True, arrowhead=3,
            ax=60, ay=-10, arrowcolor="purple",
            font=dict(size=14, color="purple")
        )
    fig_year.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig_year, use_container_width=True)

    st.markdown("### Seasonality")
    month_counts = (
        df.dropna(subset=["INSPECTION_MONTH"])
          .groupby("INSPECTION_MONTH").size().reset_index(name="count")
          .sort_values("INSPECTION_MONTH")
    )
    month_counts["MONTH"] = pd.to_datetime(month_counts["INSPECTION_MONTH"], format="%m").dt.strftime("%b")

    fig_month = px.bar(
        month_counts, x="MONTH", y="count",
        color="count", color_continuous_scale=PAL_SEQ,
        title="Inspections by Month (All Years)"
    )
    # highlight top 2 months
    top2_idx = month_counts["count"].nlargest(2).index
    fig_month.update_traces(marker_line_width=0)
    for i, bar_y in enumerate(month_counts["count"]):
        if i in top2_idx:
            fig_month.data[0].marker.color[i] = PAL_SEQ_ALT[-2]
    fig_month.update_layout(yaxis_title="Inspections", xaxis_title="")
    st.plotly_chart(fig_month, use_container_width=True)

    # Year x Month heatmap
    pivot = (
        df.dropna(subset=["INSPECTION_YEAR","INSPECTION_MONTH"])
          .groupby(["INSPECTION_YEAR","INSPECTION_MONTH"])
          .size().reset_index(name="count")
          .pivot(index="INSPECTION_YEAR", columns="INSPECTION_MONTH", values="count")
          .fillna(0)
    )
    # month labels
    pivot.columns = [pd.to_datetime(str(int(m)), format="%m").strftime("%b") for m in pivot.columns]
    fig_hm = px.imshow(
        pivot.values, x=list(pivot.columns), y=list(pivot.index.astype(int)),
        color_continuous_scale=PAL_SEQ, aspect="auto",
        title="Heatmap — Inspections by Year × Month"
    )
    fig_hm.update_layout(xaxis_title="Month", yaxis_title="Year", coloraxis_showscale=True)
    st.plotly_chart(fig_hm, use_container_width=True)

# -----------------------------
# TAB 3 — Geography
# -----------------------------
with tab3:
    st.subheader("Map — Rodent Inspections")
    # filter for valid coords
    geo = df.dropna(subset=["LATITUDE","LONGITUDE"])
    # optional borough filter
    boros = ["All"] + sorted(geo["BOROUGH"].dropna().unique().tolist())
    selected_boro = st.selectbox("Borough filter", boros, index=0)
    if selected_boro != "All":
        geo = geo[geo["BOROUGH"] == selected_boro]

    # cap points for performance
    max_points = st.slider("Max points on map", 5_000, 100_000, 20_000, 5_000)
    if len(geo) > max_points:
        geo = geo.sample(n=max_points, random_state=42)

    fig_map = px.scatter_mapbox(
        geo,
        lat="LATITUDE", lon="LONGITUDE",
        color="RESULT", color_discrete_sequence=PAL_QUAL,
        hover_data=["BOROUGH","INSPECTION_DATE","INSPECTION_TYPE","RESULT"],
        zoom=9, height=650, title="Rodent Inspections (sampled)"
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=60,b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    # NTA top 20
    st.markdown("### Top 20 Neighborhoods by Inspection Volume")
    nta_counts = (
        df["NTA"].fillna("Unknown").value_counts().head(20).reset_index()
        .rename(columns={"index":"NTA", "NTA":"count"})
        .sort_values("count", ascending=True)
    )
    fig_nta = px.bar(
        nta_counts, x="count", y="NTA", orientation="h",
        color="count", color_continuous_scale=PAL_SEQ_ALT,
        title="Top 20 NTAs"
    )
    fig_nta.update_layout(xaxis_title="Inspections", yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(fig_nta, use_container_width=True)

# -----------------------------
# TAB 4 — Quick Model (Preview)
# -----------------------------
with tab4:
    st.subheader("Predicting ‘Rat Activity’ (quick preview)")

    # Binary target: Rat Activity vs Other
    mdl = df.copy()
    mdl["target"] = (mdl["RESULT"] == "Rat Activity").astype(int)
    feat_cols = ["BOROUGH","INSPECTION_TYPE","INSPECTION_YEAR","INSPECTION_MONTH",
                 "ZIP_CODE","X_COORD","Y_COORD","LATITUDE","LONGITUDE","NTA"]
    mdl = mdl.dropna(subset=feat_cols + ["target"]).copy()

    # encode categoricals
    cat_cols = ["BOROUGH","INSPECTION_TYPE","NTA"]
    mdl[cat_cols] = mdl[cat_cols].astype("category")
    X = pd.get_dummies(mdl[feat_cols], drop_first=True)
    y = mdl["target"].values

    # Limit training rows for speed
    if len(X) > train_samples:
        X_train, _, y_train, _ = train_test_split(X, y, train_size=train_samples, random_state=42, stratify=y)
    else:
        X_train = X.copy()
        y_train = y.copy()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42, stratify=y
    )

    # Scale numeric features (mainly for Logistic Regression)
    num_cols = ["ZIP_CODE","INSPECTION_YEAR","INSPECTION_MONTH","X_COORD","Y_COORD","LATITUDE","LONGITUDE"]
    scaler = None
    if use_scaler or model_name == "Logistic Regression":
        scaler = StandardScaler()
        X_train_full[num_cols] = scaler.fit_transform(X_train_full[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Train
    if model_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=rf_trees, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        )
    else:
        clf = LogisticRegression(
            max_iter=300, n_jobs=-1, class_weight="balanced", multi_class="auto"
        )

    with st.spinner(f"Training {model_name}…"):
        clf.fit(X_train_full, y_train_full)

    # Predict & metrics
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["No Rat Activity","Rat Activity"], output_dict=True)
    rep_df = pd.DataFrame(report).T.round(3)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Accuracy", f"{report['accuracy']:.3f}")
    with m2:
        st.metric("Precision (Rat Activity)", f"{report['Rat Activity']['precision']:.3f}")
    with m3:
        st.metric("Recall (Rat Activity)", f"{report['Rat Activity']['recall']:.3f}")

    st.markdown("**Classification report**")
    st.dataframe(rep_df, use_container_width=True)

    # Confusion matrix (Plotly heatmap)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    fig_cm = px.imshow(
        cm, text_auto=True, color_continuous_scale=PAL_SEQ, aspect="auto",
        labels=dict(x="Predicted", y="True", color="Count"),
        x=["No Rat Activity","Rat Activity"], y=["No Rat Activity","Rat Activity"],
        title=f"{model_name} — Confusion Matrix"
    )
    fig_cm.update_traces(textfont_size=14)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Feature importances (if RF)
    if model_name == "Random Forest":
        importances = pd.DataFrame({
            "feature": X.columns,
            "importance": clf.feature_importances_
        }).sort_values("importance", ascending=False).head(20)
        st.markdown("**Top features (Random Forest)**")
        fig_imp = px.bar(
            importances, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale=PAL_SEQ_ALT,
            title="Top Feature Importances"
        )
        fig_imp.update_layout(xaxis_title="Importance", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")
st.caption("Data: NYC Open Data — DOHMH Rodent Inspection (2010–2024). Dashboard for course project.")
