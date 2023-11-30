import openai
import pandas as pd
import ast, re
from openai.embeddings_utils import get_embedding
from app.learning_model.helpers import normalize_text, define_recommendation_categories
from app.learning_model.pre_trained_models import encode_image, get_answer
from tenacity import RetryError
import time

from app.learning_model.messaging import (
    DETERMINE_USER_REQUEST_INSTRUCTIONS, 
    DETERMINE_USER_REQUEST_EXAMPLES, 
    DETERMINE_USER_REQUEST_INSTRUCTIONS_JSON_EXAMPLE,
    EXTEND_DATASET_REQUEST,
)
from app.learning_model.constants import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    OPENAI_API_TYPE,
    OPENAI_API_VERSION,
    GPT_ENGINE,
    TEXT_EMBEDDING_ENGINE,
    IMAGES_PATH
)

openai.api_type = OPENAI_API_TYPE
openai.api_version = OPENAI_API_VERSION
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT


def extract_features(description):
    response = openai.ChatCompletion.create(
      engine=GPT_ENGINE,
      messages=[

          {"role": "system",
          "content": normalize_text(EXTEND_DATASET_REQUEST)},

          {"role": "user",
          "content": description}
          ])['choices'][0]['message']['content']

    response = '{'+re.split('[{}]', response)[1]+'}'

    # covering the case where model returns irrelevant json
    try:
      feature_dict = ast.literal_eval(response)
    except Exception:
      print(response)
      return extract_features(description)

    return feature_dict

def custom_embedding(input_entity, task):
    assert task in ['text', 'image']
    if task == 'text':
        normalized_text = normalize_text(input_entity)
        try:
            result = get_embedding(normalized_text, engine = TEXT_EMBEDDING_ENGINE)
        except RetryError as err:
            # Azure Tier 0 has limit    ed RPS
            # That's why we need to put on hold when receive RetryError
            time.sleep(1)
            result = get_embedding(normalized_text, engine = TEXT_EMBEDDING_ENGINE)
    elif task == 'image':
        result = encode_image(input_entity)
    return result
        

def define_input_img_category(
    image_description: str, 
    df: pd.DataFrame
) -> str:
    response = openai.ChatCompletion.create(
        engine="azure-gpt-35-turbo",
        messages=[
            {
                "role": "system",
                "content": '''Instructions:
                - You are an assistant designed to classify the type 
                of the given garment to one of the 9 categories: ''' +
                ', '.join(list(df.product_group_name.unique()))
            },
            {
                "role": "user",
                "content": df[df.product_group_name == 'Swimwear'].iloc[0].detail_desc
            },
            {
                "role": "assistant",
                "content": 'Swimwear'
            },
            {
                "role": "user",
                "content": df[df.product_group_name == 'Garment Lower body'].iloc[0].detail_desc
            },
            {
                "role": "assistant",
                "content": 'Garment Lower body'
            },
            {
                "role": "user",
                "content": image_description
            }
        ]
    )
    return response['choices'][0]['message']['content']

async def determine_user_request(
    image_description: str, 
    message: str,
    category: str
) -> str:
    categories_str = define_recommendation_categories(category)
    response = await openai.ChatCompletion.acreate(
        engine=GPT_ENGINE,
        messages=[
            {
                'role': 'system',
                'content': normalize_text(
                    DETERMINE_USER_REQUEST_INSTRUCTIONS.format(
                        possible_categories=categories_str,
                        format_to_return=DETERMINE_USER_REQUEST_INSTRUCTIONS_JSON_EXAMPLE

                    ),
                ),
            },
            {
                'role': 'user',
                'content': normalize_text(DETERMINE_USER_REQUEST_EXAMPLES[0]['user'])
            },
            {
                'role': 'assistant',
                'content': normalize_text(DETERMINE_USER_REQUEST_EXAMPLES[0]['assistant'])
            },
            {
                'role': 'user',
                'content': normalize_text(DETERMINE_USER_REQUEST_EXAMPLES[1]['user'])
            },
            {
                'role': 'assistant',
                'content': normalize_text(DETERMINE_USER_REQUEST_EXAMPLES[1]['assistant'])
            },
            {
                'role': 'user',
                'content': normalize_text(DETERMINE_USER_REQUEST_EXAMPLES[2]['user'])
            },
            {
                'role': 'assistant',
                'content': normalize_text(DETERMINE_USER_REQUEST_EXAMPLES[2]['assistant'])
            },
            {
                'role': 'user',
                'content': normalize_text(DETERMINE_USER_REQUEST_EXAMPLES[3]['user'])
            },
            {
                'role': 'assistant',
                'content': normalize_text(DETERMINE_USER_REQUEST_EXAMPLES[3]['assistant'])
            },
            {
                'role': 'user',
                'content':  f'Image: {image_description}. Message: {message}'
            }
        ]
    )

    response = response['choices'][0]['message']['content']
    response = normalize_text(response).replace(';', '').replace("''", "'")
    if response.count('}') == 2:
        response = response.replace('}', '},', 1)

    return ast.literal_eval(response)
