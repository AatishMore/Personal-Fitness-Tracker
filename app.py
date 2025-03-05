import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import time

st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")

st.markdown(
    """
    <style>
        .main { background-color: #1E1E1E; padding: 20px; }
        .header { text-align: center; font-size: 34px; color: #FF6F61; font-weight: bold; }
        .subheader { text-align: center; font-size: 18px; color: #F8F9FA; margin-bottom: 20px; }
        .dashboard-container { display: flex; gap: 20px; }
        .box { background-color: #2C2C2C; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px #FF6F61; width: 100%; color: #F8F9FA; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .fade-in { animation: fadeIn 1.5s ease-in; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<p class='header fade-in'>Personal Fitness Dashboard</p>", unsafe_allow_html=True)
st.markdown("<p class='subheader fade-in' style='text-align: center; font-size: 20px; color: #FFFFFF; font-weight: bold; background-color: #FF6F61; padding: 10px; border-radius: 5px;'>Track your workouts and progress effectively.</p>", unsafe_allow_html=True)

st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    bmi = st.sidebar.slider("BMI: ", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min): ", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }
    return pd.DataFrame(data_model, index=[0])

df = user_input_features()
st.write("<div class='box fade-in'><h3>Your Parameters:</h3></div>", unsafe_allow_html=True)
st.write(df)


calories = pd.read_csv("../data/calories.csv")
exercise = pd.read_csv("../data/exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
exercise_df = exercise_df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_df = pd.get_dummies(exercise_df, drop_first=True)

X = exercise_df.drop("Calories", axis=1)
y = exercise_df["Calories"]


model = RandomForestRegressor(n_estimators=100, max_depth=6)
model.fit(X, y)


df = df.reindex(columns=X.columns, fill_value=0)
prediction = model.predict(df)

st.write("<div class='box fade-in'><h3>Predicted Calories Burned:</h3></div>", unsafe_allow_html=True)
st.write(f"<h2 style='color: #FF6F61;'>{round(prediction[0], 2)} kilocalories</h2>", unsafe_allow_html=True)


st.write("<div class='box fade-in'><h3>Similar Results:</h3></div>", unsafe_allow_html=True)
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write(similar_data.sample(5))


st.write("<div class='box fade-in'><h3>Weekly Activity Review:</h3></div>", unsafe_allow_html=True)
fig, ax = plt.subplots()
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
progress_data = np.random.randint(200, 700, size=7)
sns.barplot(x=days, y=progress_data, ax=ax, palette="coolwarm")
ax.set_ylabel("Calories Burned")
st.pyplot(fig)


st.write("<div class='box fade-in'><h3>General Information:</h3></div>", unsafe_allow_html=True)
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.write("You are older than", round(sum(boolean_age) / len(boolean_age), 2) * 100, "% of other people.")
st.write("Your exercise duration is higher than", round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "% of other people.")
st.write("You have a higher heart rate than", round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "% of other people during exercise.")
st.write("You have a higher body temperature than", round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100, "% of other people during exercise.")


st.write("<div class='box fade-in'><h3>Overall Weekly Goal Progress:</h3></div>", unsafe_allow_html=True)
goal_progress = np.random.randint(50, 100)
st.progress(goal_progress / 100)
st.write(f"You have completed **{goal_progress}%** of your weekly goal!")


# Create Virtual Environment - python -m venv myenv
# myenv\Scripts\activate
# Install Required Libraries
# pip install pandas numpy matplotlib seaborn scikit-learn streamlit jupyter


