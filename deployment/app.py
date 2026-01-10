import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ----------------------------------------
# Load saved objects
# ----------------------------------------

with open('deployment/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('deployment/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('deployment/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

with open('deployment/cuisine_features.pkl', 'rb') as f:
    cuisine_features = pickle.load(f)

with open('deployment/collection_features.pkl', 'rb') as f:
    collection_features = pickle.load(f)

# ----------------------------------------
# App UI
# ----------------------------------------

st.set_page_config(page_title="Zomato Rating Predictor", page_icon="🍽️")
st.title("🍽️ Zomato Restaurant Rating Predictor")
st.write("Predict restaurant rating based on review and metadata")

review_text = st.text_area("Enter review text")
cost = st.number_input("Average cost (₹)", min_value=0)
pictures = st.number_input("Number of pictures uploaded", min_value=0)
num_reviews = st.number_input("Reviewer past reviews", min_value=0)
num_followers = st.number_input("Reviewer followers", min_value=0)

cuisines_input = st.multiselect(
    "Select cuisines",
    [c.replace("Cuisines_", "") for c in cuisine_features]
)

collections_input = st.multiselect(
    "Select collections",
    [c.replace("Collections_", "") for c in collection_features]
)

# ----------------------------------------
# Prediction function
# ----------------------------------------

def predict_rating(review_text, cost, pictures, num_reviews, num_followers, cuisines, collections):

    input_df = pd.DataFrame({
        'Pictures': [pictures],
        'Review_Length': [len(review_text)],
        'Cost_log': [np.log1p(cost)],
        'Reviewer_Score_log': [np.log1p(0.7*num_reviews + 0.3*num_followers)],
        'Engagement_Score': [0.7*num_reviews + 0.3*num_followers],
        'Has_Pictures': [1 if pictures > 0 else 0]
    })

    for col in cuisine_features:
        input_df[col] = 1 if col.replace("Cuisines_", "") in cuisines else 0

    for col in collection_features:
        input_df[col] = 1 if col.replace("Collections_", "") in collections else 0

    input_df = input_df[feature_columns]
    input_scaled = scaler.transform(input_df)

    return model.predict(input_scaled)[0]

# ----------------------------------------
# Button action
# ----------------------------------------

if st.button("Predict Rating"):
    if review_text.strip() == "":
        st.warning("Please enter a review text.")
    else:
        pred = predict_rating(
            review_text,
            cost,
            pictures,
            num_reviews,
            num_followers,
            cuisines_input,
            collections_input
        )

        st.success(f"⭐ Predicted Rating: {round(pred, 2)}")

