from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ExifTags
from translate import Translator
from transformers import DonutProcessor, VisionEncoderDecoderModel
import sentencepiece as spm
import torch

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
    # Load the processor and the model
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    print(pixel_values, " pixel values1")

    # Check the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pixel_values = pixel_values.to(device)
    print(pixel_values, " pixel values2")

    # Perform the prediction
    outputs = model.generate(pixel_values, max_length=512, num_beams=5, early_stopping=True)
    print(outputs)
    predicted_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return predicted_text