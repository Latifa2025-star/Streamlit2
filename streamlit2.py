# streamlit2.py
# NYC DOHMH Rodent Inspections — Interactive Dashboard (2010–2024)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------
# Streamlit page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NYC Rodent Inspections — Interactive Dashboard",
    page_icon="🐀",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Color palettes (color-blind friendly)
# -----------------------------------------------------------------------------
PAL_SEQ = px.colors.sequential.OrRd
PAL_SEQ_ALT = px.colors.sequential.Sunset
PAL_QUAL = px.colors.qualitative.Set2

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.title("Controls")
st.sidebar.caption("Filter, sampling, and modeling options")

sample_n = st.sidebar.number_input(
    "Sample rows (0 = all — may be slow)", min_value=0, max_value=2_000_000,
    value=250_000, step=50_000,
    help="Use a sample to keep the app snappy. Set to 0 to load the full dataset."
)

with st.sidebar.expander("Modeling options (quick preview)", expanded=False):
    model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
    train_size = st.slider("Train fraction", 0.50, 0.95, 0.80, 0.05)
    rf_trees = st.number_input("Random Forest trees", 50, 500, 200, 50)
    scale_numeric = st.checkbox("Scale numeric features (helps Logistic Regression)", True)
    train_rows_cap = st.number_input("Max training rows", 50_000, 500_000, 150_000, 25_000)

# -----------------------------------------------------------------------------
# Data loading & cleaning
# -----------------------------------------------------------------------------
DATA_URL = "https://data.cityofnewyork.us/api/views/p937-wjvj/rows.csv?accessType=DOWNLOAD"

@st.cache_data(show_spinner=True)
def load_data(url: str, sample_n: int = 0) -> pd.DataFrame:
    df = pd.read_csv(url, low_memory=False)
    if sample_n and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=42)
    return df

@st.cache_data(show_spinner=True)
def clean_enrich(df: pd.DataFrame) -> pd.DataFrame:
    # Parse dates
    df["INSPECTION_DATE"] = pd.to_datetime(df["INSPECTION_DATE"], errors="coerce")
    df["APPROVED_DATE"] = pd.to_datetime(df["APPROVED_DATE"], errors="coerce")

    # Time window per proposal
    df = df[(df["INSPECTION_DATE"] >= "2010-01-01") & (df["INSPECTION_DATE"] <= "2024-12-31")]

    # Date parts
    df["INSPECTION_YEAR"] = df["INSPECTION_DATE"].dt.year
    df["INSPECTION_MONTH"] = df["INSPECTION_DATE"].dt.month

    # Minimal cleaning
    df = df.dropna(subset=["BOROUGH", "RESULT"])

    # Normalize labels (guard against odd variants)
    df["RESULT"] = df["RESULT"].replace({
        "Bait applied": "Bait applied",
        "Rat Activity": "Rat Activity",
        "Passed": "Passed",
        "Failed for Other R": "Failed for Other R",
        "Monitoring visit": "Monitoring visit"
    })

    return df

with st.spinner("Loading NYC DOHMH Rodent Inspection data…"):
    raw = load_data(DATA_URL, sample_n=sample_n)
    df = clean_enrich(raw)

# -----------------------------------------------------------------------------
# Header & KPIs
# -----------------------------------------------------------------------------
st.title("🐀 NYC DOHMH Rodent Inspections — 2010–2024")
st.caption("Interactive EDA, geovisuals, and quick baseline modeling. Built with Streamlit + Plotly.")

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Rows", f"{len(df):,}")
with k2: st.metric("Columns", f"{df.shape[1]:,}")
with k3:
    yr_min, yr_max = int(df["INSPECTION_YEAR"].min()), int(df["INSPECTION_YEAR"].max())
    st.metric("Year range", f"{yr_min}–{yr_max}")
with k4: st.metric("Boroughs", df["BOROUGH"].nunique())

st.markdown("---")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_time, tab_geo, tab_model = st.tabs(
    ["Overview", "Time & Seasonality", "Geography", "Model (Preview)"]
)

# -----------------------------------------------------------------------------
# TAB: Overview
# -----------------------------------------------------------------------------
with tab_overview:
    st.subheader("Outcome Mix & Borough Distribution")
    cA, cB = st.columns(2)

    with cA:
        outcome_counts = (
            df["RESULT"].fillna("Unknown")
              .value_counts().rename_axis("RESULT").reset_index(name="count")
        )
        fig_pie = px.pie(
            outcome_counts, names="RESULT", values="count",
            hole=0.45, color="RESULT", color_discrete_sequence=PAL_QUAL,
            title="Inspection Outcomes Mix"
        )
        fig_pie.update_traces(textinfo="percent+label", textposition="inside")
        st.plotly_chart(fig_pie, use_container_width=True)

    with cB:
        borough_counts = (
            df["BOROUGH"].value_counts().rename_axis("BOROUGH").reset_index(name="count")
        )
        fig_b = px.bar(
            borough_counts, x="BOROUGH", y="count",
            color="count", color_continuous_scale=PAL_SEQ_ALT,
            title="Inspections by Borough"
        )
        fig_b.update_layout(xaxis_title="", yaxis_title="Inspections", coloraxis_showscale=False)
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown(
        "- **Passed** dominates overall.\n"
        "- **Rat Activity** is a sizable minority and varies by borough.\n"
        "- Highest volumes occur in **Manhattan**, **Brooklyn**, and **Bronx**."
    )

# -----------------------------------------------------------------------------
# TAB: Time & Seasonality
# -----------------------------------------------------------------------------
with tab_time:
    st.subheader("Trends over Time")

    # Yearly trend
    year_counts = (
        df.groupby("INSPECTION_YEAR").size().reset_index(name="count").sort_values("INSPECTION_YEAR")
    )
    fig_year = px.line(
        year_counts, x="INSPECTION_YEAR", y="count", markers=True,
        color_discrete_sequence=[PAL_SEQ[4]],
        title="Inspections per Year (2010–2024)"
    )
    # COVID-19 dip annotation (inside the plot area, left of 2020)
    if 2020 in year_counts["INSPECTION_YEAR"].values:
        y2020 = year_counts.loc[year_counts["INSPECTION_YEAR"] == 2020, "count"].values[0]
        fig_year.add_annotation(
            x=2019.4, y=y2020 * 1.06,
            text="<b>COVID-19 dip</b>", showarrow=True, arrowhead=3,
            ax=60, ay=-10, arrowcolor="purple",
            font=dict(size=14, color="purple")
        )
    fig_year.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig_year, use_container_width=True)

    st.markdown("### Seasonality (All Years Combined)")
    month_counts = (
        df.dropna(subset=["INSPECTION_MONTH"])
          .groupby("INSPECTION_MONTH").size().reset_index(name="count")
          .sort_values("INSPECTION_MONTH")
    )
    month_counts["MONTH"] = pd.to_datetime(
        month_counts["INSPECTION_MONTH"], format="%m"
    ).dt.strftime("%b")

    fig_month = px.bar(
        month_counts, x="MONTH", y="count",
        color="count", color_continuous_scale=PAL_SEQ,
        title="Inspections by Month"
    )
    fig_month.update_layout(yaxis_title="Inspections", xaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(fig_month, use_container_width=True)

    # Year × Month heatmap
    pivot = (
        df.dropna(subset=["INSPECTION_YEAR", "INSPECTION_MONTH"])
          .groupby(["INSPECTION_YEAR", "INSPECTION_MONTH"]).size().reset_index(name="count")
          .pivot(index="INSPECTION_YEAR", columns="INSPECTION_MONTH", values="count")
          .fillna(0)
    )
    pivot.columns = [pd.to_datetime(str(int(m)), format="%m").strftime("%b") for m in pivot.columns]

    fig_hm = px.imshow(
        pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index.astype(int)),
        color_continuous_scale=PAL_SEQ,
        aspect="auto",
        title="Heatmap — Inspections by Year × Month"
    )
    fig_hm.update_layout(xaxis_title="Month", yaxis_title="Year")
    st.plotly_chart(fig_hm, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB: Geography
# -----------------------------------------------------------------------------
with tab_geo:
    st.subheader("Map — Rodent Inspections")

    geo = df.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
    boro_opts = ["All"] + sorted(geo["BOROUGH"].dropna().unique().tolist())
    sel_boro = st.selectbox("Borough filter", boro_opts, index=0)
    if sel_boro != "All":
        geo = geo[geo["BOROUGH"] == sel_boro]

    max_points = st.slider("Max points on map (sampled for speed)", 5_000, 100_000, 20_000, 5_000)
    if len(geo) > max_points:
        geo = geo.sample(n=max_points, random_state=42)

    fig_map = px.scatter_mapbox(
        geo,
        lat="LATITUDE", lon="LONGITUDE",
        color="RESULT", color_discrete_sequence=PAL_QUAL,
        hover_data=["BOROUGH", "INSPECTION_DATE", "INSPECTION_TYPE", "RESULT"],
        zoom=9, height=650, title="Rodent Inspections (sampled)"
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=60, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("### Top 20 Neighborhoods (NTA) by Volume")
    nta_counts = (
        df["NTA"].fillna("Unknown").value_counts().head(20).reset_index()
          .rename(columns={"index": "NTA", "NTA": "count"})
          .sort_values("count", ascending=True)
    )
    fig_nta = px.bar(
        nta_counts, x="count", y="NTA", orientation="h",
        color="count", color_continuous_scale=PAL_SEQ_ALT,
        title="Top 20 NTAs"
    )
    fig_nta.update_layout(xaxis_title="Inspections", yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(fig_nta, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB: Modeling (quick preview; binary target Rat Activity vs Other)
# -----------------------------------------------------------------------------
with tab_model:
    st.subheader("Predicting ‘Rat Activity’ (Quick Preview)")

    mdl = df.copy()
    mdl["target"] = (mdl["RESULT"] == "Rat Activity").astype(int)

    feature_cols = [
        "BOROUGH", "INSPECTION_TYPE", "INSPECTION_YEAR", "INSPECTION_MONTH",
        "ZIP_CODE", "X_COORD", "Y_COORD", "LATITUDE", "LONGITUDE", "NTA"
    ]
    mdl = mdl.dropna(subset=feature_cols + ["target"]).copy()

    # Encode categoricals
    cat_cols = ["BOROUGH", "INSPECTION_TYPE", "NTA"]
    for c in cat_cols:
        mdl[c] = mdl[c].astype("category")

    X = pd.get_dummies(mdl[feature_cols], drop_first=True)
    y = mdl["target"].values

    # Cap training rows to keep the demo responsive
    if len(X) > train_rows_cap:
        X_small, _, y_small, _ = train_test_split(
            X, y, train_size=train_rows_cap, stratify=y, random_state=42
        )
    else:
        X_small, y_small = X.copy(), y.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_small, y_small, train_size=train_size, stratify=y_small, random_state=42
    )

    # Scale numeric columns when desired (helps LR; RF unaffected)
    numeric_cols = ["ZIP_CODE", "INSPECTION_YEAR", "INSPECTION_MONTH",
                    "X_COORD", "Y_COORD", "LATITUDE", "LONGITUDE"]
    if scale_numeric or model_choice == "Logistic Regression":
        scaler = StandardScaler()
        for _df in (X_train, X_test):
            for col in numeric_cols:
                if col in _df.columns:
                    # fit on train only
                    pass
        X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Train model
    if model_choice == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=int(rf_trees), random_state=42, n_jobs=-1,
            class_weight="balanced_subsample"
        )
    else:
        # Avoid n_jobs (not available in some sklearn versions on Streamlit Cloud)
        clf = LogisticRegression(
            max_iter=400, class_weight="balanced", multi_class="auto"
        )

    with st.spinner(f"Training {model_choice}…"):
        clf.fit(X_train, y_train)

    # Predict & metrics
    y_pred = clf.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=["No Rat Activity", "Rat Activity"],
        output_dict=True,
        zero_division=0
    )
    rep_df = pd.DataFrame(report).T.round(3)

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Accuracy", f"{report['accuracy']:.3f}")
    with m2: st.metric("Precision (Rat Activity)", f"{report['Rat Activity']['precision']:.3f}")
    with m3: st.metric("Recall (Rat Activity)", f"{report['Rat Activity']['recall']:.3f}")

    st.markdown("**Classification report**")
    st.dataframe(rep_df, use_container_width=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig_cm = px.imshow(
        cm, text_auto=True, color_continuous_scale=PAL_SEQ, aspect="auto",
        labels=dict(x="Predicted", y="True", color="Count"),
        x=["No Rat Activity", "Rat Activity"], y=["No Rat Activity", "Rat Activity"],
        title=f"{model_choice} — Confusion Matrix"
    )
    fig_cm.update_traces(textfont_size=14)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Feature importance (RF only)
    if model_choice == "Random Forest":
        importances = (
            pd.DataFrame({"feature": X_train.columns, "importance": clf.feature_importances_})
              .sort_values("importance", ascending=False)
              .head(20)
        )
        st.markdown("**Top Features (Random Forest)**")
        fig_imp = px.bar(
            importances, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale=PAL_SEQ_ALT,
            title="Top Feature Importances"
        )
        fig_imp.update_layout(xaxis_title="Importance", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")
st.caption("Data: NYC Open Data — DOHMH Rodent Inspection (2010–2024).")
