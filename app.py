import streamlit as st
import requests
import base64
from PIL import Image
import io
import cv2 as cv
import numpy as np
from utils import pyramid_blending,webp_to_np
# Function to resize the image to allowed dimensions
def resize_image(image, allowed_dimensions):
    # Find the closest allowed dimensions
    closest_dim = min(allowed_dimensions, key=lambda x: abs(x[0] - image.width) + abs(x[1] - image.height))
    return image.resize(closest_dim)

# Function to call the Stable Diffusion API
def generate_image(init_image, text_prompt):
    response = requests.post(
        "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer sk-vx9ZiBMTTcwWR85CI9W46OXTVeCXLG7zO63gsdkNR24BxJy3"
        },
        files={
            "init_image": init_image
        },
        data={
            "init_image_mode": "IMAGE_STRENGTH",
            "image_strength": 0.35,
            "steps": 40,
            "seed": 0,
            "cfg_scale": 5,
            "samples": 1,
            "text_prompts[0][text]": text_prompt,
            "text_prompts[0][weight]": 1,
        }
    )
    return response

# Streamlit app
st.title("Public Art Demo")

# Load local images
image1 = cv.imread("sd1.jpg")
resized_image1=cv.resize(image1, (1024, 1024))
image2 = cv.imread("sd2.jpg")
resized_image2=cv.resize(image2, (1024, 1024))
 
if 'current_image1' not in st.session_state:
    st.session_state['current_image1'] = resized_image1
if 'current_image2' not in st.session_state:
    st.session_state['current_image2'] = resized_image2
if 'previous_image1' not in st.session_state:
    st.session_state['previous_image1'] = resized_image1
if 'previous_image2' not in st.session_state:
    st.session_state['previous_image2'] = resized_image2

# Display the local images and input prompts
col1, col2 = st.columns(2)
with col1:
    text_prompt1 = st.text_input("Enter a prompt for image 1:", "A painting of a cat", key="prompt1")
    if st.button("Generate Image 1"):
        # Process image 1
        # allowed_dimensions = [(1024, 1024), (1152, 896), (1216, 832), (1344, 768), (1536, 640),
        #                       (640, 1536), (768, 1344), (832, 1216), (896, 1152)]
        # resized_image1 = resize_image(st.session_state['current_image1'], allowed_dimensions)
        # init_image_bytes1 = io.BytesIO()
        # resized_image1.save(init_image_bytes1, format=st.session_state['current_image1'].format)
        # init_image_bytes1 = io.BytesIO(init_image_bytes1.getvalue())
        _, buffer = cv.imencode('.jpg', st.session_state['current_image1'])
        init_image_bytes1 = io.BytesIO(buffer)
        response1 = generate_image(init_image_bytes1, text_prompt1)
        if response1.status_code == 200:
            data1 = response1.json()
            for img in data1["artifacts"]:
                generated_image1 = base64.b64decode(img["base64"])
                generated_image1 = np.array(Image.open(io.BytesIO(generated_image1)))
                st.session_state['previous_image1'] = st.session_state['current_image1']
                st.session_state['current_image1'] = generated_image1
        else:
            st.error(f"Error: {response1.text}")

with col2:
    text_prompt2 = st.text_input("Enter a prompt for image 2:", "A painting of a dog", key="prompt2")
    if st.button("Generate Image 2"):
        # Process image 2
        # allowed_dimensions = [(1024, 1024), (1152, 896), (1216, 832), (1344, 768), (1536, 640),
        #                       (640, 1536), (768, 1344), (832, 1216), (896, 1152)]
        # resized_image2 = resize_image(st.session_state['current_image2'], allowed_dimensions)
        # init_image_bytes2 = io.BytesIO()
        # resized_image2.save(init_image_bytes2, format=st.session_state['current_image2'].format)
        # init_image_bytes2 = io.BytesIO(init_image_bytes2.getvalue())
        _, buffer = cv.imencode('.jpg', st.session_state['current_image2'])
        init_image_bytes2 = io.BytesIO(buffer)
        response2 = generate_image(init_image_bytes2, text_prompt2)
        if response2.status_code == 200:
            data2 = response2.json()
            for img in data2["artifacts"]:
                generated_image2 = base64.b64decode(img["base64"])
                generated_image2 = np.array(Image.open(io.BytesIO(generated_image2)))
                st.session_state['previous_image2'] = st.session_state['current_image2']
                st.session_state['current_image2'] = generated_image2
        else:
            st.error(f"Error: {response2.text}")

# Display the grid of four images

concat_previous_image=pyramid_blending(st.session_state['previous_image1'],st.session_state['previous_image2'])
concat_current_image=pyramid_blending(st.session_state['current_image1'],st.session_state['current_image2'])
st.write("### Previous Images")
prev_col1, prev_col2, prev_col3 = st.columns([1,1,2])
with prev_col1:
    st.image(st.session_state['previous_image1'], caption="Previous Image 1", use_column_width=True)
with prev_col2:
    st.image(st.session_state['previous_image2'], caption="Previous Image 2", use_column_width=True)
with prev_col3:
    st.image(concat_previous_image, caption="Previous Concat", use_column_width=True)

st.write("### Current Images")
curr_col1, curr_col2,curr_col3 = st.columns([1,1,2])
with curr_col1:
    st.image(st.session_state['current_image1'], caption="Current Image 1", use_column_width=True)
with curr_col2:
    st.image(st.session_state['current_image2'], caption="Current Image 2", use_column_width=True)
with curr_col3:
    st.image(concat_current_image, caption="Current concat", use_column_width=True)