# Digital Genius Tech Task
Create a Product Recommendation engine based on product images and a single customer message. Incorporate an understanding of images as well as text for multi- modal input. Focus on understanding/processing images and text. You can use Azure OpenAI access for text understanding if that suits your solution.  
This repository contains code decision for the recommendation task for e-commerce.

**Requirements**:

- Input: image and text → Output: product recommendation
- Understanding of customer messages using NLP/Multi-Modal ML techniques (e.g: LLM prompting, multi-modal models) for product recommendation input
- Find similar items based on text + image
- Find complementary items based on text + image


# Code

`app/datasets` folder contains 2 files: 
- articles.csv is the original dataset of garments in the stock
- extended_articles.csv contains preprocessed data from articles.csv. It is extended with material type, seasonality and occasion features as well as saved text and image embeddings

## Set up

1. Install requirements:
`pip install requirements.txt`

2. Fill in credentials (Azure API Key) in `app/learning_model/constants.py` file

3. Run server 
command: `uvicorn app.main:api`

5. Send python request (I used Postman) and fill custom image and message:
```
import requests
url = "http://localhost:8000/prompt?text={MESSAGE}"
payload = {}  
files=[
  ('item',('test_image.jpg',open('{IMAGE_PATH}','rb'), 'image/jpeg'))
]
headers = {}
response = requests.request("POST", url, headers=headers, data=payload, files=files)
```

## Data preprocessing - extending the dataset

To improve recommendations based on text descriptions it's important to extract valuable features and not include the noise.
For this purpose I've decided to use gpt model to extract from image descriptions important features like material type, seasonality and occasion.
I will use them to combine in more consice descriptions which will be better for finding embeddings similarities.

## Underdstanding user's input
I have used `gpt-3.5-turbo` assistant to understand user's request and classify it into one of 2 tasks - either find specific garment or find complementary outfit.
In the second case assistant also gives recommendations based on complementary categories, which will be used to find garments in the stock.

## Embeddings

I have used `ada-002` to get text embeddings and `clip-ViT-B-32` from Sentence Transformers to encode images.
Text embeddings were performed on normalized text.

## Finding similarities

I have used cosine similarity for both image and text embeddings. I have also tried other metrics for embedding similarity, such as euclidean distance, manhattan distance, minkowski distance and jaccard similarity but they didn't show better performance.

What about embeddings I have chosen, I compared image embeddings, text embeddings and combined both image and text embeddings. The best result was shown when using image embeddings.

I have included 3 approaches in the code - comparing only images, only texts or combining both methods. In case of combining I have taken common sum of both embeddings.

## Further improvement

The code functionality can be improved by extracting the number of garment from the input message either asking customer how many garments should be shown and then give this number as a parameter to the search_items function from the app/learning_model/recommendations.py file.
Both models for text-to-image and image-to-text tasks can be fine-tuned with external data from online shops.
For the task of looking for complementarities the images for the embeddings were generated by generative network so we can't completely rely on them although anyway they consist important patterns. Probably this is not the best approach and can be improved by combining image and text embeddings in a 1-vector space but the text preprocessing and tokenization needs to be considered.
In my decision very poor attention given to colors however it's one of the most important feature while looking for complementary garments. I tried to do it straightforward by looking for similar colors but the task in non-trivial because it's not the rule that similar colors should go well. Accordingly to the color combination rule the complementary colors are three adjacent colors on the color wheel. After some research I have found https://www.thecolorapi.com which proposes api to suggest color schemes (complementary colors) for the specific color given in the hex-code format. I think this approach deserves considering and can significantly improve recommendations.

This decision is not straighforward. I have used some pre-trained models to extend dataset, to generate text descriptions and images. This code combines all of these methods and gives pretty nice result.The good option would be to gather customer's feedback in order to use it later as labels to train collaborative filtering system. Also personilised customer's profile and history would be helpful.

