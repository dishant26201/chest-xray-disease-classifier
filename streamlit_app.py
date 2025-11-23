# app/streamlit_app.py

import streamlit as st
from PIL import Image
import pandas as pd
import tensorflow as tf
from src.preprocessing import predict_labels
from src.config import MODEL_PATH

#  Take the raw dict from predict_labels and return a list sorted by probability in descending order
def format_predictions(preds_dict):

    items = list(preds_dict.items()) # Convert dict to list of (label, prob) pairs

    sorted_preds = sorted(items, key=lambda x: x[1], reverse=True) # Sort by probability in descending order

    return sorted_preds

# Load the chest X-ray classifier model from disk and cache
@st.cache_resource
def load_chest_xray_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False) # Load the model from the .h5 file
    return model


def main():
    st.title("Chest X-ray Disease Classifier") # Page title
    
    st.subheader("EXPERIMENTAL DEMO (NOT FOR CLINICAL USE)") # DISCLAIMER FOR USER AS THIS ISN'T FOR CLINICAL USE

    # Brief description about the tool
    st.markdown(
    """

    This is an experimental machine learning model which was trained on frontal chest radiographs (NIH ChestX-ray14). 
    While it can identify patterns associated with common chest x-rays, it is **not a diagnostic tool** and may produce incorrect results.
    Please do not upload other image types (e.g., photos, lateral x-rays, etc) as the model will produce invalid predictions.

    ##### **Model reliability notice**

    This model is still a work in progress:
    - It has been trained for a limited number of epochs.  
    - Further training and tuning are required to improve performance.  
    - Longer training requires additional compute resources which is a limitation.

    Always interpret predictions cautiously, and never use this model for clinical or decision-making purposes.
    """
)

    # File uploader widget to input image
    uploaded_file = st.file_uploader(
        "Upload a Frontal Chest X-ray image (PNG or JPG)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file) # Open image with PIL
        except Exception:
            st.error("Could not open the uploaded file as an image.") # Error if unable to open file
            return

        st.image(image, width=320, caption="Uploaded image") # Show the image back to the user


        model = load_chest_xray_model() # Load the model

        # Run model prediction along with a spinner
        with st.spinner("Running model..."):
            preds_dict = predict_labels(image, model)

        sorted_preds = format_predictions(preds_dict) # Format predictions

        threshold = 0.5  # Threshold to classify a label as "prediction"

        # Make list of conditions where probability >= threshold
        predicted_conditions = [(label, prob) for label, prob in sorted_preds if prob >= threshold]

        # If no conditions were predicted display message
        if len(predicted_conditions) == 0:
            st.markdown(
                "**Predicted condition:** None"
                f"(All probabilities are below the {threshold:.3f} threshold)."
            )
        else:
            st.markdown(
                f"<div style='font-size:20px;'><b>Predicted condition:</b> {predicted_conditions[0][0]}</div>",
                unsafe_allow_html=True
            ) # Display the predicted condition (bold the side label)

        # Spaces
        st.write("")
        st.write("")
        st.write("")

        probs_df = pd.DataFrame(sorted_preds, columns=["Condition", "Probability"]) # Create a DataFrame for all labels and probabilities
        chart_df = probs_df.set_index("Condition") # Set index to condition names

        col1, col2 = st.columns(2)  # Initialise two columns

        # Column 1: Probability table
        with col1:
            st.markdown("#### Probability Table")
            st.dataframe(probs_df)

        # Column 2: Bar chart with probabilities
        with col2:
            st.markdown("#### Probability Distribution Chart")
            st.bar_chart(chart_df)


if __name__ == "__main__":
    main()