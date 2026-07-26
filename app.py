import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

# ── Load data & train model ─────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("data/student_data.csv")

    X = df.drop("Final_Score", axis=1)
    y = df["Final_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R2": r2_score(y_test, y_pred),
    }

    return df, model, metrics


df, model, metrics = load_and_train()

# ── Title ─────────────────────────────────────────────────────
st.title("🎓 Student Performance Predictor")
st.write("Predict a student's final score based on study habits.")

# ── Sidebar: model performance ──────────────────────────────
with st.sidebar:
    st.header("Model Performance")
    st.metric("R² Score", f"{metrics['R2']:.4f}")
    st.metric("MAE", f"{metrics['MAE']:.2f}")
    st.metric("RMSE", f"{metrics['RMSE']:.2f}")
    st.caption("Trained on a small 10-row sample dataset — treat these numbers as illustrative rather than robust.")

# ── Input form ────────────────────────────────────────────────
st.subheader("Enter Student Details")

col1, col2 = st.columns(2)
with col1:
    study_hours = st.slider("Study Hours (per day)", 1, 10, 5)
    attendance = st.slider("Attendance (%)", 50, 100, 75)
with col2:
    previous_score = st.slider("Previous Score", 40, 90, 70)
    sleep_hours = st.slider("Sleep Hours (per day)", 5, 9, 7)

if st.button("🔍 Predict Final Score", type="primary"):
    input_data = pd.DataFrame([{
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Score": previous_score,
        "Sleep_Hours": sleep_hours,
    }])

    prediction = model.predict(input_data)[0]

    st.divider()
    st.metric("Predicted Final Score", f"{prediction:.2f}")
    st.progress(min(max(prediction / 100, 0.0), 1.0))

# ── Visualization: effect of study hours ────────────────────
st.divider()
st.subheader("Effect of Study Hours on Final Score")
st.caption(f"Holding Attendance={attendance}%, Previous Score={previous_score}, Sleep={sleep_hours}h constant")

hours_range = list(range(1, 11))
predictions = [
    model.predict(pd.DataFrame([{
        "Study_Hours": h,
        "Attendance": attendance,
        "Previous_Score": previous_score,
        "Sleep_Hours": sleep_hours,
    }]))[0]
    for h in hours_range
]

fig, ax = plt.subplots()
ax.plot(hours_range, predictions, marker="o")
ax.set_xlabel("Study Hours")
ax.set_ylabel("Predicted Final Score")
ax.set_title("Study Hours vs Predicted Score")
st.pyplot(fig)

# ── Raw data ──────────────────────────────────────────────────
with st.expander("View training data"):
    st.dataframe(df)