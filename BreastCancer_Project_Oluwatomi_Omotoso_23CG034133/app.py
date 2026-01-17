import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models
model = joblib.load("./models/logistic_regression_model.pkl")
scaler = joblib.load("./models/scaler.pkl")

# Titles
st.title("🏥 Breast Cancer Prediction Tool")
st.write("Enter patient measurements below:")

labels_1 = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "symmetry_mean",
]
(
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean,
    smoothness_mean,
    compactness_mean,
    concavity_mean,
    symmetry_mean,
) = labels_1

features = []
col1_items = [
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean,
    smoothness_mean,
    compactness_mean,
    concavity_mean,
    symmetry_mean,
]

for item in col1_items:
    item = st.number_input(item, min_value=0.000000)
    features.append(item)

if st.button("Predict"):
    # Collect all inputs
    user_data = np.array(features).reshape(1, -1)
    scaled_data = scaler.transform(user_data)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0]

    # Predictions and probabilities.
    if prediction[0] == 1:
        st.error(f"Prediction: MALIGNANT")
    else:
        st.success(f"Prediction: BENIGN")

    # 5. Create the Visualization Data
    prob_df = pd.DataFrame(
        {
            "Diagnosis": ["Bening", "Malignant"],
            "Probability": probability,
        }
    )

    # 6. Display the Comparison Bar Chart
    st.markdown("### Probability Breakdown")
    # Use st.bar_chart for a quick, native Streamlit plot
    st.bar_chart(prob_df.set_index("Diagnosis"), use_container_width=True)

    # Optional: Display the raw probabilities
    st.write("Raw Probabilities:", probability)
