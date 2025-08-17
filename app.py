#teachable
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Class labels from labels.txt
CLASS_NAMES = [
    "Pituitary Tumor",
    "No Tumor",
    "Meningioma Tumor",
    "Glioma Tumor"
]

# Load TFLite model
@st.cache_resource
def load_tflite_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading TFLite model: {e}")
        return None

def preprocess_image(image, target_size, channels):
    image = image.resize(target_size)
    if channels == 1:  # Grayscale
        image = image.convert("L")  # Convert to grayscale
    elif channels == 3:  # RGB
        image = image.convert("RGB")  # Convert to RGB

    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
    if channels == 1:  # For grayscale, add the channel dimension
        image_array = np.expand_dims(image_array, axis=-1)  # Shape: (height, width, 1)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension: (1, height, width, channels)
    return image_array


# Make predictions with the TFLite model
def predict(interpreter, image_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Streamlit App
st.title("Brain Tumor Classification")
st.text("Upload an MRI scan to classify.")

# Upload an image
uploaded_image = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the TFLite model
    model_path = "model.tflite"  # Replace with the path to your model file
    interpreter = load_tflite_model(model_path)
    if interpreter:
        # Check input shape of the model
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']  # e.g., (1, 96, 96, 1) or (1, 96, 96, 3)
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]
        st.write(f"Model expects input shape: {height}x{width} with {channels} channel(s).")
        
        # Preprocess the image
        processed_image = preprocess_image(image, target_size=(height, width), channels=channels)

        # Make predictions
        predictions = predict(interpreter, processed_image)
        class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        # Display results
        st.write(f"Predicted Class: {CLASS_NAMES[class_index]}")
        st.write(f"Confidence: {confidence:.2f}")

        # Show probabilities as a bar chart
        st.bar_chart(predictions[0])
    else:
        st.error("Failed to load the TFLite model.")
