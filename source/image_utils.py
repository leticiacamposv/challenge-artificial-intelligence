from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ExifTags
from translate import Translator
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

def extract_text_from_image(image_path, custom_oem_psm_config=None):
    """
    Page segmentation modes: (PSM)
    0    Orientation and script detection (OSD) only.
    1    Automatic page segmentation with OSD.
    2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
    3    Fully automatic page segmentation, but no OSD. (Default)
    4    Assume a single column of text of variable sizes.
    5    Assume a single uniform block of vertically aligned text.
    6    Assume a single uniform block of text.
    7    Treat the image as a single text line.
    8    Treat the image as a single word.
    9    Treat the image as a single word in a circle.
    10    Treat the image as a single character.
    11    Sparse text. Find as much text as possible in no particular order.
    12    Sparse text with OSD.
    13    Raw line. Treat the image as a single text line,
        bypassing hacks that are Tesseract-specific.
    
    OCR Engine modes: (OEM)
    0    Legacy engine only.
    1    Neural nets LSTM engine only.
    2    Legacy + LSTM engines.
    3    Default, based on what is available.
  """
    img = Image.open(image_path)
    
    # OCR
    if custom_oem_psm_config:
        text = pytesseract.image_to_string(img, lang='por', config=custom_oem_psm_config)  # Language code for Portuguese
    else:
        text = pytesseract.image_to_string(img, lang='por')
    return text

