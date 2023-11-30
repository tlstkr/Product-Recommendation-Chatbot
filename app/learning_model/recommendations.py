import pandas as pd
from openai.embeddings_utils import cosine_similarity

from app.learning_model.constants import IMAGES_PATH
from app.learning_model.dataset_parser import read_embeddings
from app.learning_model.openai_client import define_input_img_category, custom_embedding
from app.learning_model.helpers import normalize_text
from app.learning_model.pre_trained_models import generate_image

def find_similarities(input_emb, embds_dict):
  similarity_dict = {}
  for article, emb in embds_dict.items():
    similarity_dict[article] = cosine_similarity(input_emb, emb)
  return similarity_dict
# list(sorted(similarity_dict.items(), key=lambda x:x[1], reverse=True))

def get_recommended_indexes(recommendation_dict, threshold, top_n):
  dataset = pd.DataFrame({'article': recommendation_dict.keys(), 'similarity': recommendation_dict.values()})
  best = dataset[dataset.similarity > threshold]
  sorted_df = dataset.sort_values("similarity", ascending=False)
  result = best.head(top_n) if len(best) > 0 else sorted_df.head(1)
  return list(result.article)

def find_similar_garment(df, input_entity, category, task, threshold=0.9, top_n=3) -> list:
  assert(task in ['image', 'text'])
  
  # input entity here is an image_path in case of searching by images 
  # and garment_description in case of searching by text descriptions 

  # get embedding depending on task (image or text)
  embedding = custom_embedding(input_entity, task)      
  stack_embds = read_embeddings(df[df.product_group_name == category], f'{task}_embeddings')
  # get dict of recommendations in format {'article':'similarity'}
  recommendations_dict = find_similarities(embedding, stack_embds)
  # transforming to list of articles taking the best matches
  recommended_articles = get_recommended_indexes(recommendations_dict, threshold, top_n)
  return recommended_articles


def find_complementarities(df, list_of_garment_descriptions, task):
  recommended_articles = []
  for garment_description in list_of_garment_descriptions:
    garment_description = normalize_text(garment_description)
    category = define_input_img_category(image_description=garment_description, df=df)
    if task == 'text':
        recommended_articles.append(find_similar_garment(df, garment_description, category, 'text', threshold=0.9, top_n=3))

    elif task == 'image':
        print(f'generating image for {garment_description}')
        image = generate_image(garment_description)
        generic_image_path = f'{IMAGES_PATH}generated_garment.jpg'
        image.save(generic_image_path)
        recommended_articles.append(find_similar_garment(df, generic_image_path, category, 'image', threshold=0.9, top_n=3))

    elif task == 'both':
        image = generate_image(garment_description)
        generic_image_path = f'{IMAGES_PATH}generated_garment.jpg'
        image.save(generic_image_path)
        image_embedding = custom_embedding(generic_image_path, 'image')      
        image_stack_embds = read_embeddings(df[df.product_group_name == category], f'image_embeddings')
        text_embedding = custom_embedding(garment_description, 'text')      
        text_stack_embds = read_embeddings(df[df.product_group_name == category], f'image_embeddings')
        
        image_similarities = find_similarities(image_embedding, image_stack_embds)
        text_similarities = find_similarities(text_embedding, text_stack_embds)

        combined_similarities = {}

        for article, image_similarity in image_similarities.items:
          combined_similarities[article] = image_similarity + text_similarities[article]

        recommended_articles = get_recommended_indexes(combined_similarities, 0.9, 3)

       
  return recommended_articles


def get_result(df:pd.DataFrame, recommendations: dict, user_image_path:str=None, input_garment_category:str=None) -> list:
    if recommendations['task'] == 'find':
        print(f"Looking for the {recommendations['garment']}")

        # get list of recommended articles
        recommended_articles = find_similar_garment(df, user_image_path, input_garment_category, 'image', threshold=0.9, top_n=3)

    if recommendations['task'] == 'complement':
        print(f'Trying to find something which would fit good, for example {", ".join(recommendations["garments"])}')

        # approach 1 - by text
        recommended_articles = find_complementarities(df, recommendations['garments'], 'text')

        # approach 2 - by image
        recommended_articles = find_complementarities(df, recommendations['garments'], 'image')

        # combined approach
        find_complementarities(df, recommendations['garments'], 'both')
        

    return recommended_articles