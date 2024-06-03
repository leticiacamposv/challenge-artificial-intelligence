from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ExifTags
from translate import Translator
from transformers import DonutProcessor, VisionEncoderDecoderModel
import sentencepiece as spm
import torch
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\letic\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

def get_image_description(image_path, model_name="Salesforce/blip-image-captioning-large"):
    # Load the processor and model
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    image = Image.open(image_path).convert("RGB")

    # Pre-processamento
    inputs = processor(images=image, return_tensors="pt")

    # Predict
    outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)

    # Pos-processamento
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    translator = Translator(to_lang="pt")
    translated_caption = translator.translate(caption)

    return translated_caption

def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    if exif_data:
        return {ExifTags.TAGS.get(tag): value for tag, value in exif_data.items()}
    return {}

def extract_text_from_image(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(img, lang='por')  # Language code for Portuguese
    
    return text

# Path to your image
image_path = '../resources/Infografico-1.jpg'

# Extract text from image
extracted_text = extract_text_from_image(image_path)

print("Extracted Text:", extracted_text)
