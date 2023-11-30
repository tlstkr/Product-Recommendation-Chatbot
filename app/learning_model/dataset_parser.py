import os
import numpy as np

import pandas as pd
from app.learning_model.openai_client import custom_embedding, extract_features
from app.learning_model.pre_trained_models import generate_image, get_image_description, get_answer
from app.learning_model.constants import IMAGES_PATH


def get_custom_embeddings(df: pd.DataFrame, task: str) -> pd.DataFrame:
    print(f'Generating {task} embeddings started: ')
    result = {}

    for idx, article in enumerate(df.index):
        
        if (idx+1) % 25 == 0:
            print(f'{idx+1} / {len(df.index)}')
        
        if task == 'text':
            input_entity = f'{df.product_type_name[article]} in {df.colour_group_name[article]} {df.material[article]}. {df.seasonality[article]} {df.occasion[article]}'
            input_entity = df.loc[[article]].detail_desc.item()
                
        elif task == 'image':
            img_path = f'{IMAGES_PATH}0{article}.jpg'
            if not os.path.exists(img_path):
                print(f'There is no image for the item with article {article}. Generating custom one')        
                image = generate_image(df.detail_desc[article])
                image.save(img_path)
            input_entity = img_path                

        result[article] = custom_embedding(input_entity, task)
    
    return result


def extend_dataset(df):
    result = {
      'material':[],
      'seasonality':[],
      'occasion':[]}
    for article in df.index:
        # defining description as concatenating most important features
        concat_desc = f'{df.colour_group_name[article]} {df.graphical_appearance_name[article]} {df.product_type_name[article]}.  {df.detail_desc[article]}'
        feature_dict = extract_features(concat_desc)

        for key, value in result.items():
            if key in feature_dict:
                value.append(feature_dict[key])
            else:
                if key == 'material':
                    question = 'Which type of material this garment is made from?'
                elif key == 'seasonality':
                    question = 'Which season across summer, spring/fall, fall/winter and all year this garment belongs to?'
                elif key == 'occasion':
                    question = 'Which purpose I can wear it for?'

                try:
                    answer = get_answer(f'{IMAGES_PATH}0{article}.jpg', question)
                except FileNotFoundError:
                    answer = '' 
                value.append(answer)

    df['material'] = result['material']
    df['seasonality'] = result['seasonality']
    df['occasion'] = result['occasion']
    return df


def read_fashion_df() -> pd.DataFrame:

    df_path = 'app/datasets/articles.csv'
    extended_df_path = 'app/datasets/extended_articles.csv'

    try:
        df = pd.read_csv(extended_df_path, index_col='article_id')
    
    except FileNotFoundError:
        print('Preprocessed dataset has not been found. Creating the new one')
        
        # load data
        df = pd.read_csv(df_path, index_col='article_id')

        # fill na descriptions using image-to-text generator
        for article in list(df[df.detail_desc.isna()].index):
            image_path = f'{IMAGES_PATH}0{article}.jpg'
            image_description = get_image_description(image_path)
            df.detail_desc[article] = image_description
        
        # extend dataset with material, seasonality and occasion
        df = extend_dataset(df)

        # generate text embeddings
        df['text_embeddings'] = get_custom_embeddings(df, 'text')
        df.to_csv(extended_df_path)
        
        # generate missing images and image embeddings 
        df['image_embeddings'] = get_custom_embeddings(df, 'image')
        df.to_csv(extended_df_path)

    return df


def process_encoding(encoding):
    encoding_list = encoding.replace('[', '').replace(']', '').split(', ') 
    print(encoding_list)
    encoding_array = np.asarray(encoding_list, dtype = np.float32)
    return encoding_array


def read_embeddings(df, embeddings_column_name):
  emb_dict = {}
  for article in df.index:
    emb = df.loc[[article]][embeddings_column_name].item()
    if type(emb) == str:
        emb = process_encoding(emb)
    emb_dict[article] = emb
  return  emb_dict
