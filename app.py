import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from skimage.color import gray2rgb, rgb2lab, lab2rgb
from skimage.transform import resize
from skimage import img_as_ubyte
import cv2

# Load your trained model
model_path = '/content/Art_Colorization_Model.h5'
model = load_model(model_path)

# Function to create InceptionResNetV2 embedding
def create_inception_embedding(grayscaled_rgb):
    def resize_gray(x):
        return resize(x, (299, 299, 3), mode='constant')

    grayscaled_rgb_resized = np.array([resize_gray(x) for x in grayscaled_rgb])
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)

    # Use the model directly without the 'graph' attribute
    embed = inception.predict(grayscaled_rgb_resized)

    return embed

# Function to colorize the image
def colorize_image(image):
    # Ensure that the image has values between 0 and 255
    image = np.clip(image, 0, 255)

    # Normalize pixel values to the range [0, 1]
    img = image / 255.0

    # Preprocess the input image
    img = resize(img, (256, 256, 3), mode='constant')
    img = img.reshape((1, 256, 256, 3))

    # Preprocess the input image for InceptionResNetV2
    color_me_embed = create_inception_embedding(gray2rgb(rgb2lab(img)[:, :, 0]))

    # Colorize the image using the trained model
    output = model.predict([img, color_me_embed])
    output = output * 128

    # Initialize the decoded image array
    decoded_img = np.zeros((1, 256, 256, 3))

    # Create the LAB image
    cur = np.zeros((256, 256, 3))
    cur[:, :, 0] = img[0][:, :, 0]
    cur[:, :, 1:] = resize(output[0], (256, 256, 2), mode='constant')

    # Convert LAB to RGB
    decoded_img[0] = lab2rgb(cur) * 255
    return decoded_img[0].astype(np.uint8)

def main():
    st.title("Image Colorization App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            # Read the image
            image = np.array(Image.open(uploaded_file))

            # Colorize the image
            colorized_image = colorize_image(image)

            # Display the colorized image
            st.image(colorized_image, caption="Colorized Image.", use_column_width=True)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
