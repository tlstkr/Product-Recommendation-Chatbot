import re

def normalize_text(s: str) -> str:
   s = re.sub(r'\s+',  ' ', str(s)).strip()
   s = re.sub(r". ,","",s)
   s = s.replace("..",".")
   s = s.replace(". .",".")
   s = s.replace("\n", "")
   s = s.strip()
   return s


def define_recommendation_categories(category):
  separated_look = ['Garment Upper body', 'Garment Lower body', 'Socks & Tights', 'Shoes', 'Accessories']
  full_look = ['Garment Full body',
 'Socks & Tights', 'Shoes', 'Accessories']
  sleep_look = ['Nightwear', 'Socks & Tights']
  summer_look = ['Swimwear', 'Accessories']

  recommendations = []

  for group in [separated_look, full_look, sleep_look, summer_look]:
    if category in group:
      recommendations.extend(group)

  recommendations = list(set(recommendations))
  recommendations.remove(category)

  return ', '.join(recommendations)
