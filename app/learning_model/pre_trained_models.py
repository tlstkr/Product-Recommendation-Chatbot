import PIL
import torch

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

from app.learning_model.constants import IMAGE_TO_TEXT_TASK, BLIP_IMAGE_TO_TEXT_MODEL, CLIP_SENTENCE_TRANSFORMER, STABLE_DIFFUSOR, VQA_MODEL, VQA_TASK


def load_stable_diffusor():
    scheduler = EulerDiscreteScheduler.from_pretrained(STABLE_DIFFUSOR, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(STABLE_DIFFUSOR, scheduler=scheduler, torch_dtype=torch.float32)
    # pipe = pipe.to("cuda")
    return pipe

def check_models_are_downloaded_to_server() -> None:
    # Image-To-Text
    pipeline(IMAGE_TO_TEXT_TASK, BLIP_IMAGE_TO_TEXT_MODEL)
    print(f'{BLIP_IMAGE_TO_TEXT_MODEL} model has been loaded')
    # Image Encoder
    SentenceTransformer(CLIP_SENTENCE_TRANSFORMER)
    print(f'{CLIP_SENTENCE_TRANSFORMER} model has been loaded')
    # Text-To-Image
    load_stable_diffusor()
    print(f'{STABLE_DIFFUSOR} model has been loaded')
    # Visual Question Answering
    pipeline(VQA_TASK, model=VQA_MODEL)

def get_image_description(image_path: str) -> str:
    """ Get image description from saleforce model """
    model_blip_img_capt = pipeline(IMAGE_TO_TEXT_TASK, BLIP_IMAGE_TO_TEXT_MODEL)
    response = model_blip_img_capt(image_path)[0]['generated_text']
    return response.replace('a close up of a', '')

def encode_image(image_path: str):
  """ Model from Sentence Transformer to encode images """
  model_clip = SentenceTransformer(CLIP_SENTENCE_TRANSFORMER)
  embedding = model_clip.encode(PIL.Image.open(image_path))
  return embedding

def generate_image(description: str):
    """ Generate image from description using stable diffusor """
    pipe = load_stable_diffusor()
    response = pipe(description).images[0]
    return response

def get_answer(image_path, question):
    blip_vqa = pipeline(VQA_TASK, model=VQA_MODEL)
    answer = blip_vqa(image_path, question, return_tensors="pt")[0]['answer']
    return answer

