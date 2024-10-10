import PIL.Image
from transformers import pipeline
import base64
import PIL
from langchain_core.documents import Document


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def img2txt(path: str):
    img = PIL.Image.open(path)

    image_to_text = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-large", device="cuda", max_new_tokens=50
    )

    res = image_to_text(img)

    return res


def txt2doc(text: str, path: str, modality: str):
    metadata = {
        "source": path,
        "type": modality,
    }
    document = Document(page_content=text, metadata=metadata)

    return document
