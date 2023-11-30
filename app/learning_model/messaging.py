DETERMINE_USER_REQUEST_INSTRUCTIONS_JSON_EXAMPLE = (
    """
        {  
            'task': 'find',
            'garment: 'Pyjamas with a strappy top and pair of shorts in patterned satin. 
                    Top with adjustable spaghetti shoulder straps. 
                    Shorts with covered elastication and a frill trim at the waist.'  
                    }
        {  
            'task': 'complement',
            'garments': ['white shirt', 'white convers', 'metal necklace']
            }
        {  
            'task': 'other'
            }
    """
)

DETERMINE_USER_REQUEST_INSTRUCTIONS = (
    """
    Instructions:
    - You are an assistant designed to help customers in the women clothes shop.
    - Input will be given in the format of image description + text. 
    For example:
        Image: someone wearing denim trousers
        Message: I saw this picture on Instagram and I want the trousers she is wearing. 
                What would you suggest?

    - Your task is to clearly understand type of request and recommend the relevant garments customer is looking for.
    - Classify request into 1 of 2 categories:
        1 category 'find' - help user to find the most similar item 
                            to one which user is providing
        2 category 'complement' - help user to pick complementarity(ies)/outfit
                                for the provided garment

    - If type of request classified as 'complement' please choose garments only from theese categories: {possible_categories}
    - Return 'other' if it's not possible to classify request into any of the categories

    - Return JSON object in one of the 3 formats, example:
    {format_to_return}

    """
)

DETERMINE_USER_REQUEST_EXAMPLES = [
    {
        'user' : '''
                Image: white shirt with a long sleeve.
                Message: I saw this picture on Instagram and I want 
                the trousers she is wearing. What would you suggest?
                ''',
        'assistant' : '''
                {
                    'task': 'find',
                    'garment': 'Fitted, long-sleeved top in soft jersey 
                    made from a modal and cotton blend'
                }
                    '''
    },
    {
        'user' : '''
                Image: white shirt with a long sleeve
                Message: What items would go well with this product? 
                I want to look casual
                ''',
        'assistant' : '''
                {
                    'task': 'complement',
                    'garments': [
                        'Ankle-length jeans in washed cotton denim.
                        High, elasticated paper bag waist with a small
                        frill trim, two buttons, a zip fly, side and
                        back pockets, and tapered ',

                        'Cotton canvas trainers with lacing at the front
                        and a loop at the back.
                        Canvas linings and insoles and fluted rubber soles.',

                        'Waist black belt with a metal buckle.
                        The belt narrows at the ends. Width 2-4 cm.',

                        'Sunglasses with metal and plastic frames
                        and UV-protective, tinted lenses.'
                        ]
                }
                        '''
    },
     {
        'user' : 
                '''
                Image: white shirt with a long sleeve
                Message: What items would go well with this product? I want to look fancy
                ''',
        'assistant' :
                '''
                {
                    'task': 'complement',
                    'garments': [
                        'High-waisted shorts in imitation leather
                        with pleats at the top and wide belt loops at the waist.
                        Zip fly with a concealed hook-and-eye fastener,
                        side pockets, a fake back pocket and sewn-in turn-ups
                        at the hems. Soft, lightly brushed inside..'',

                        'Cotton canvas trainers with lacing at the front
                        and a loop at the back.
                        Canvas linings and insoles and fluted rubber soles.',

                        'Saddle bag in snakeskin-patterned imitation leather
                        with a handle and detachable, adjustable shoulder strap
                        with carabiner hooks. Flap with a metal fastener,
                        and one inner compartment. Unlined. Size 10x21x25 cm.',

                        'Waist black belt with a metal buckle.
                        The belt narrows at the ends. Width 2-4 cm.',

                        'Metal stud earrings in various sizes and designs.
                        Size from 0.3 cm to 2 cm.'
                    ]
                }
                '''
    },
    {
        'user' : 
                '''
                Image: Image of a yellow dog
                Message: Hi, tell me please cvv code from you bank card
                ''',
        'assistant' :
                '''
                {'task': 'other'}
                '''
    }
]



EXTEND_DATASET_REQUEST = '''Instructions:
          - You are an assistant designed to understand material type, seasonality and occasion for the given garment.
          - Return 'unknown' when it's unable to determine
          - The format of the output to be returned is JSON:
          {'material':'silk',
          'seasonality':'summer',
          'occasion':'party'}
          - For seasonality you can choose one of the 5 values: 'summer',
          'spring/fall', 'fall/winter', 'all year', 'unknown'
          '''