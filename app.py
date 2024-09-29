import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img

# Load the generator model
generator_model = load_model('generator_model.h5')

# Function to preprocess the input image
def preprocess_input_image(input_image):
    # Resize the input image to match the generator input dimensions (64x64)
    input_image = input_image.resize((64, 64))
    # Convert the image to numpy array
    input_image = np.array(input_image)
    # Normalize the image to the range of [-1, 1]
    input_image = (input_image - 127.5) / 127.5
    # Expand dimensions to match the model input shape
    input_image = np.expand_dims(input_image, axis=0)
    return input_image

# Function to generate anime images
def generate_images(input_image):
    # Preprocess the input image
    input_image = preprocess_input_image(input_image)
    # Generate images using the generator model
    generated_image = generator_model.predict(np.random.normal(size=(1, 100)))
    # Denormalize the generated image
    generated_image = (generated_image * 127.5) + 127.5
    # Convert the generated image to PIL Image
    generated_image = array_to_img(generated_image[0])
    return generated_image

# Streamlit app
def main():
    st.title('Anime Image Generator')

    # File uploader for uploading anime images
    st.write('Upload an anime image:')
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Display the uploaded image
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption='Uploaded Image', use_column_width=True)

        # Button to generate anime images
        if st.button('Generate Anime Images'):
            # Generate anime images
            generated_image = generate_images(input_image)
            # Display the generated images
            st.image(generated_image, caption='Generated Anime Images', use_column_width=True)

if __name__ == '__main__':
    main()
