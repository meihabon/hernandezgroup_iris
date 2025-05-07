import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("Iris.csv")
df = df.drop(columns=["Id"])

# mapping species manually
df["Species"] = df["Species"].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
})

# Features and labels
X = df.drop(columns=["Species"])
y = df["Species"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("Iris Predictor (Linear Regression)")

sl = st.number_input("Sepal Length", min_value=0.0, step=0.1)
sw = st.number_input("Sepal Width", min_value=0.0, step=0.1)
pl = st.number_input("Petal Length", min_value=0.0, step=0.1)
pw = st.number_input("Petal Width", min_value=0.0, step=0.1)


if st.button("Classify"):
    data = [[float(sl), float(sw), float(pl), float(pw)]]
    result = model.predict(data)
    result = int(round(result[0]))

    if result == 0:
        st.write("Iris-setosa")
    elif result == 1:
        st.write("Iris-versicolor")
    else:
        st.write("Iris-virginica")
