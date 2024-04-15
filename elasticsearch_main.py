from elasticsearch import Elasticsearch
#USING THIS FOR FRONT END PART
# Initialize the Elasticsearch client
es = Elasticsearch(['https://edbrown.mids255.com/'], basic_auth=('elastic', 'jFoEj4A4444rCrQm'))


def search_recipes(es, lemmatized_ingredient_1, lemmatized_ingredient_2, lemmatized_ingredients):
    # Construct the Elasticsearch query to search for recipes based on the provided ingredient names
    # query = {
    #     "query": {
    #         "bool": {
    #             "must": [{"match": {"ingredients": ingredient}} for ingredient in ingredients]
    #         }
    #     }
    # }

    #1. Add in the boost_factors
    #2. Lemmatize the ingredients --> SEE APP-GOTMEALS.PY

    #*NEW*
    boost_factors = {
    lemmatized_ingredient_1: 3,
    lemmatized_ingredient_2: 2}

    #*NEW*
    query = {
    "query": {
        "bool": {
            "must": [
                {"match": {"cleaned_NER": {"query": lemmatized_ingredient_1, "boost": boost_factors.get(lemmatized_ingredient_1, 1)}}},
                {"match": {"cleaned_NER": {"query": lemmatized_ingredient_2, "boost": boost_factors.get(lemmatized_ingredient_2, 1)}}},  # New must clause for ingredient2
            ],
            "should": [
                {"match": {"cleaned_NER": {"query": ingredient, "boost": boost_factors.get(ingredient, 1)}}}
                for ingredient in lemmatized_ingredients if ingredient is not None
            ],

            "minimum_should_match": 1
            }
        }
    }


    # Execute the search query
    response = es.search(index="recipes", body=query)

    return response

if __name__ == "__main__":
    
    
    # Define the ingredients to search for
    ingredient1 = "beef"
    ingredient2 = "potato"
    ingredient3 = "asparagus"
    ingredient4 = "mushroom"
    ingredient5 = None
    must_not = "tomato"

    # List of optional ingredients provided by the user (could include None values)
    optional_ingredients = [ingredient2, ingredient3, ingredient4, ingredient5]

    # Construct the initial query
    query = {
        "query": {
            "bool": {
                "must": [{"match": {"ingredients": ingredient1}}],
                "should": [{"match": {"ingredients": ingredient}} for ingredient in optional_ingredients if ingredient is not None],
                "must_not": [{"match": {"ingredients": must_not}}],
                "minimum_should_match": 1
            }
        }
    }

    # Execute the search with fallback
    result = search_recipes(es, query)
    print(result)

