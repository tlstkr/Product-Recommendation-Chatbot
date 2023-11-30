import aiofiles
import PIL

from fastapi import FastAPI, UploadFile
from fastapi.responses import Response

from app.learning_model.constants import IMAGES_PATH
from app.learning_model.pre_trained_models import get_image_description, check_models_are_downloaded_to_server
from app.learning_model.dataset_parser import read_fashion_df
from app.learning_model.recommendations import get_result
from app.learning_model.openai_client import define_input_img_category, determine_user_request

api = FastAPI()

check_models_are_downloaded_to_server()


@api.get('/healthcheck')
async def healthcheck() -> Response:
    return {'status': 'OK'}


@api.post('/prompt')
async def prompt(text: str, item: UploadFile) -> Response:
    user_image_path = f'{IMAGES_PATH}user_image.jpg'
    async with aiofiles.open(user_image_path, 'wb') as out_file:
        content = await item.read()
        await out_file.write(content)

    df = read_fashion_df()
    print('df has been loaded')
    # Generate input image description using Salesforce/blip-image-captioning-large
    image_description = get_image_description(user_image_path)

    # Define which category garment from the input image belongs to using gpt
    input_garment_category = define_input_img_category(
        image_description=image_description, 
        df=df,
    )

    # Classify user's request either to 'find' or 'complement' task
    # and get the list of recommended garments descriptions using gpt
    recommendations = await determine_user_request(
        image_description=image_description, 
        message=text,
        category=input_garment_category,
    )

    recommendations = (
        [recommendations]
        if isinstance(recommendations, dict)
        else recommendations
    )

    recommended_articles = []
    for recommendation in recommendations:
        recommended_articles.extend(get_result(df, recommendation, user_image_path, input_garment_category))

    images = {}
    for article in recommended_articles:
        desc = df.detail_desc[article]
        images[desc] = f'{IMAGES_PATH}0{article}.jpg'
        PIL.Image.open(images[desc]).show()
    
    return images
