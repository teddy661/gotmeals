##0. Import FastAPI and ElasticSearch
##1. Search the recipe index using ElasticSearch 
##2. Return the result
##3. Post the result to localhost/search

from modeles import es

ingredient1 = "beef"
ingredient2 = "potato"
ingredient3 = "asparagus"
ingredient4 = "mushroom"
ingredient5 = None
must_not = "tomato"


# List of optional ingredients provided by the user (could include None values)
optional_ingredients = [ingredient2, ingredient3, ingredient4, ingredient5] 

def search_recipes(es, query):
    response = es.search(index="recipes", body=query, size=5)
    if not response['hits']['hits']:  # Check if the query returned no results
        # Modify the query for a broader search
        # For example, remove the `minimum_should_match` or adjust `must_not` clause
        # This is a placeholder for how you might adjust the query
        query['query']['bool'].pop('minimum_should_match', None)  # Example adjustment
        response = es.search(index="recipes", body=query, size=10)  # Try again with the adjusted query
    return response

# Initial query
query = {
    "query": {
        "bool": {
            "must": [
                {"match": {"ingredients": ingredient1}}
            ],
            "should": [
                {"match": {"ingredients": ingredient}}
                for ingredient in optional_ingredients if ingredient is not None
            ],
            "must_not": [
                {"match": {"ingredients": must_not}},
            ],
            "minimum_should_match": 1
        }
    }
}

# Execute search with fallback
result = search_recipes(es, query)
print(result)

#Pull in ingredients from Streamlit Front End
#We need to be able to POST the response to Streamlit Front End