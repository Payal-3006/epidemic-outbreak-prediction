import os
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Epidemic Insight | Outbreak Prediction",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #102238; }
    [data-testid="stSidebar"] * { color: #f3f6fb; }
    .hero {
        padding: 1.35rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(115deg, #102238 0%, #155e75 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { color: white; margin: 0; font-size: 2.15rem; }
    .hero p { color: #d8e8f0; margin: .45rem 0 0 0; font-size: 1rem; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e3e9f1;
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(20, 40, 70, .04);
    }
    .section-note { color: #526174; font-size: .92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
FEATURE_NAMES = ["lag_1", "lag_2", "lag_3", "rolling_mean_3"]


@st.cache_resource
def load_saved_model():
    """Load a trained pipeline if one has been saved in the project."""
    candidates = [
        Path("pkl/model.dill"),
        Path("pkl/model.pkl"),
        Path("model.dill"),
        Path("model.pkl"),
    ]
    for model_path in candidates:
        if model_path.exists():
            with model_path.open("rb") as model_file:
                return dill.load(model_file), str(model_path)
    return None, None


def make_history(case_yesterday, case_two_days, case_three_days):
    """Build a small dataframe from the three user-provided observations."""
    return pd.DataFrame(
        {
            "Day": ["3 days ago", "2 days ago", "Yesterday"],
            "Cases": [case_three_days, case_two_days, case_yesterday],
        }
    )


def calculate_demo_estimate(values):
    """
    Transparent fallback when no trained model artifact is available.
    This is a heuristic demo estimate, NOT a trained ML prediction.
    """
    oldest, middle, latest = values
    mean_value = float(np.mean(values))
    rising = latest > middle > oldest
    falling = latest < middle < oldest

    if mean_value == 0:
        risk_label, risk_score, explanation = (
            "Low activity",
            10,
            "No cases were entered for the three observations.",
        )
    elif rising:
        risk_label, risk_score, explanation = (
            "Rising trend",
            75,
            "The entered case values increase at each observation.",
        )
    elif latest > mean_value:
        risk_label, risk_score, explanation = (
            "Elevated activity",
            60,
            "The latest value is above the three-observation average.",
        )
    elif falling:
        risk_label, risk_score, explanation = (
            "Declining trend",
            30,
            "The entered case values decrease at each observation.",
        )
    else:
        risk_label, risk_score, explanation = (
            "Stable / mixed trend",
            45,
            "The entered values do not show a strictly rising or falling pattern.",
        )
    return risk_label, risk_score, explanation


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🦠 Epidemic Insight")
    st.caption("Outbreak trend analysis dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Dashboard", "Make a Prediction", "Project Details"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("### Data input")
    st.caption("Enter non-negative case counts for three recent observations.")
    cases_yesterday = st.number_input(
        "Cases yesterday", min_value=0, value=10, step=1
    )
    cases_two_days = st.number_input(
        "Cases 2 days ago", min_value=0, value=8, step=1
    )
    cases_three_days = st.number_input(
        "Cases 3 days ago", min_value=0, value=5, step=1
    )
    run_prediction = st.button("Analyze case trend", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Educational demonstration • Not a medical or public-health diagnosis")

# Keep the values chronological for charts.
history = make_history(cases_yesterday, cases_two_days, cases_three_days)
case_values = [cases_three_days, cases_two_days, cases_yesterday]
rolling_mean = float(np.mean(case_values))
change = cases_yesterday - cases_two_days
change_pct = (
    (change / cases_two_days) * 100 if cases_two_days != 0 else None
)
model, model_path = load_saved_model()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🦠 Epidemic Outbreak Prediction</h1>
        <p>Explore recent case trends, compare observations, and review a prediction result.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.warning(
        "No saved trained model was found. Charts use the values entered in the sidebar. "
        "Predictions are clearly labelled as a simple demonstration heuristic, not an ML model output."
    )
else:
    st.success(f"Loaded saved model artifact: `{model_path}`")

# ---------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------
if page == "Dashboard":
    st.subheader("At a glance")
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Latest cases", f"{cases_yesterday:,}", delta=f"{change:+,} vs. 2 days ago")
    metric2.metric("3-observation average", f"{rolling_mean:.1f}")
    metric3.metric(
        "Latest change",
        f"{change_pct:+.1f}%" if change_pct is not None else "N/A",
        help="Percentage change compared with the value entered for 2 days ago.",
    )
    metric4.metric(
        "Trend direction",
        "Rising" if cases_yesterday > cases_two_days else (
            "Falling" if cases_yesterday < cases_two_days else "Unchanged"
        ),
    )

    st.markdown("### Case trend visualizations")
    left, right = st.columns([1.45, 1])

    with left:
        st.markdown("#### Recent case counts")
        chart_df = history.set_index("Day")
        st.line_chart(chart_df, height=320, use_container_width=True)
        st.caption("Line chart based only on the three values entered in the sidebar.")

    with right:
        st.markdown("#### Compare observations")
        st.bar_chart(chart_df, height=320, use_container_width=True)
        st.caption("Bar chart makes differences between the entered observations easier to compare.")

    st.markdown("### Summary")
    summary_left, summary_right = st.columns(2)
    with summary_left:
        st.markdown("#### Case count distribution")
        st.area_chart(chart_df, height=230, use_container_width=True)
    with summary_right:
        st.markdown("#### Input data")
        display_df = history.copy()
        display_df["Change from previous observation"] = display_df["Cases"].diff().fillna(0).astype(int)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.info(
            "These are user-entered example values, not live surveillance data. "
            "The chart does not establish whether an actual epidemic is occurring."
        )

# ---------------------------------------------------------
# Prediction page
# ---------------------------------------------------------
elif page == "Make a Prediction":
    st.subheader("Analyze recent observations")
    st.markdown(
        '<p class="section-note">The model expects four features: three lag values and their rolling mean.</p>',
        unsafe_allow_html=True,
    )

    feature_col, result_col = st.columns([1, 1.2])
    with feature_col:
        st.markdown("#### Feature values")
        feature_df = pd.DataFrame(
            {
                "Feature": FEATURE_NAMES,
                "Value": [
                    cases_yesterday,
                    cases_two_days,
                    cases_three_days,
                    rolling_mean,
                ],
            }
        )
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
        st.caption(
            "The rolling mean is calculated from the three entered observations. "
            "Keep the input order consistent with the model's training features."
        )

    with result_col:
        st.markdown("#### Result")
        if run_prediction:
            input_array = np.array(
                [[cases_yesterday, cases_two_days, cases_three_days, rolling_mean]],
                dtype=float,
            )
            if model is not None:
                try:
                    prediction = int(model.predict(input_array)[0])
                    if prediction == 1:
                        st.error("Model output: Class 1")
                        st.markdown("**Interpretation:** The loaded model classified this input as the positive class.")
                    else:
                        st.success("Model output: Class 0")
                        st.markdown("**Interpretation:** The loaded model classified this input as the negative class.")

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(input_array)[0]
                        probability_df = pd.DataFrame(
                            {"Class": [str(c) for c in model.classes_], "Probability": probabilities}
                        ).set_index("Class")
                        st.markdown("##### Model class probabilities")
                        st.bar_chart(probability_df, use_container_width=True)
                except Exception as exc:
                    st.error(f"Could not generate a prediction from the saved model: {exc}")
                    st.caption("Check that the saved model expects exactly these four features.")
            else:
                label, score, explanation = calculate_demo_estimate(case_values)
                st.metric("Demonstration trend indicator", label)
                st.progress(score / 100)
                st.caption(f"Trend indicator: {score}/100 — this is a heuristic visualization, not a calibrated probability.")
                st.write(explanation)
                st.info(
                    "For genuine ML inference, train and save the fitted pipeline from Model_final.py "
                    "as `pkl/model.dill`, then restart the app."
                )
        else:
            st.info("Enter case values in the sidebar and click **Analyze case trend** to see a result.")
            st.markdown("##### Trend preview")
            st.line_chart(history.set_index("Day"), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Observation-to-observation change")
    change_df = history.copy()
    change_df["Change"] = change_df["Cases"].diff().fillna(0).astype(int)
    st.bar_chart(change_df.set_index("Day")[["Change"]], use_container_width=True)

# ---------------------------------------------------------
# Project information page
# ---------------------------------------------------------
else:
    st.subheader("About this project")
    st.write(
        "This application is a frontend for exploring recent case-count inputs and demonstrating "
        "how a prediction interface can present model results. The current project repository uses "
        "lag features, a three-observation rolling mean, and a Random Forest classification pipeline "
        "in its training script."
    )

    st.markdown("### Machine learning workflow")
    workflow = pd.DataFrame(
        {
            "Stage": [
                "Data preprocessing",
                "Feature engineering",
                "Model training",
                "Model evaluation",
                "Application",
            ],
            "Purpose": [
                "Clean and organize historical observations",
                "Create lag values and a rolling mean",
                "Fit a classification pipeline",
                "Review classification metrics and ROC curve",
                "Present input values, charts, and model output",
            ],
        }
    )
    st.dataframe(workflow, use_container_width=True, hide_index=True)

    st.markdown("### Model limitations to keep in mind")
    st.markdown(
        """
        - A reliable forecast requires a clearly defined target and appropriate future-oriented evaluation.
        - The source preprocessing script reads COVID-19 case data but renames the target column to `zika_cases`; this naming should be corrected.
        - The app can load a saved fitted pipeline from `pkl/model.dill` or `pkl/model.pkl`. If no artifact is found, it uses a clearly labelled heuristic demo instead.
        - The current charts visualize the values entered by the user; they are not a live epidemiological data feed.
        - This educational project must not be used to make real-world medical or public-health decisions.
        """
    )

    st.markdown("### Expected input features")
    st.code(
        "['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3']",
        language="python",
    )

st.markdown("---")
st.caption(
    "Epidemic Insight • Student project interface • Predictions and charts are for educational demonstration only."
)
