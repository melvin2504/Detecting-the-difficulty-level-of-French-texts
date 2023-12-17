import streamlit as st
from model_predictor import FrenchSentenceDifficultyPredictor

model_path = r'E:/hec masters/DSML/cefr_predictor/model/model_for_stremlitApp.pkl'
label_encoder_path = r'E:/hec masters/DSML/cefr_predictor/model/label_encoderforApp.pkl'

# Create an instance of the predictor
predictor = FrenchSentenceDifficultyPredictor(model_path, label_encoder_path)

st.title("French Sentence Difficulty Predictor")

# User input
user_input = st.text_area("Enter a French sentence:", height=150)

# Button for predictions
if st.button("Predict Difficulty"):
    # Get predictions
    difficulty, scores = predictor.predict_difficulty(user_input)  # Modify your method to return both label and scores

    # Display the predicted difficulty level
    st.write(f"The predicted difficulty level is: {difficulty}")

    # Display scores for all categories if available
    if scores:
        st.write("Scores for each difficulty level:")
        for label, score in scores.items():  # Assuming scores is a dict with {label: score}
            st.write(f"{label}: {score}")
