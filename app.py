import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_generator(image, num_captions):
    num_captions = int(float(num_captions))
    raw_image = image.convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(
        **inputs,
        num_return_sequences=num_captions,
        max_length=64,
        early_stopping=True,
        num_beams=num_captions,
        no_repeat_ngram_size=2,
        length_penalty=0.8
    )
    captions = ""
    for i, caption in enumerate(out):
        captions += processor.decode(caption, skip_special_tokens=True) + "\n"
    return captions

# Create a Streamlit interface
st.title("OpeninApp Image Caption Generation Assignment")
photo = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])
num_captions = st.selectbox("Select number of captions to generate", [1, 2, 3, 4, 5])
caption_button = st.button("Generate Captions")
caption_output = st.empty()

# Define the caption generation function to be called when the button is clicked
def generate_captions():
    if photo is not None:
        image = Image.open(io.BytesIO(photo.read()))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        captions = caption_generator(image, num_captions)
        caption_output.text(captions)

# Call the caption generation function when the button is clicked
if caption_button:
    generate_captions()
st.write('<p style="font:bold;font-size:35px; text-align:center">Made By Smit Patel</p>',unsafe_allow_html=True)