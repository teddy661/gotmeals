##Goal of this file is to: 
##1. Load ElasticSearch into Port 9200
##2. load csv file
##3. Create the mapping and recipe index needed for Elastic Search

##BEFORE RUNNING THIS FILE YOU MUST RUN:

## python -m venv .venv
## source .venv/bin/activate
## python -m pip install pandas==1.4.3 notebook==6.3.0 elasticsearch==8.7.0
## docker run --rm -p 9200:9200 -p 9300:9300 -e "xpack.security.enabled=false" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.7.0

## pip install fastapi
## pip install uvicorn[standard]

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd
from pathlib import Path 

es = Elasticsearch(['https://edbrown.mids255.com/'], basic_auth=('elastic', 'jFoEj4A4444rCrQm'))

#Load the data
#UPDATE THE PATH FOR WHERE YOU ARE STORING THE RECIPE CSV
csv_file = Path("C:/Users/melha/Documents/ds210/project/recipe_dataset_cleaned_v4.csv")
if not csv_file.exists():
    csv_file = Path("C:/Users/teddy/Documents/01-Berkeley/210/gotmeals/EDA/recipe_dataset_cleaned_v4.csv")
if not csv_file.exists():
    csv_file = Path("C:/Users/edbrown/Documents/01-Berkeley/210/gotmeals/EDA/recipe_dataset_cleaned_v4.csv")
if not csv_file.exists():
    raise FileNotFoundError("File not found")
df = (pd.read_csv(csv_file, engine = 'python')
     .dropna()
     .sample(5000, random_state=42)
     .reset_index()
     )

#Create an index
mappings = {
        "properties": {
            "index": {"type": "integer"},
            "title": {"type": "text", "analyzer": "english"},
            "ingredients": {"type": "text", "analyzer": "standard"},
            "directions": {"type": "text", "analyzer": "standard"},
            "link": {"type": "keyword"},
            "source": {"type": "text", "analyzer": "standard"},
            "NER": {"type": "text", "analyzer": "english"},
            "cleaned_NER": {"type": "text", "analyzer": "english"},
    }
}

# #Delete an existing index
# es.indices.delete(index='recipes')

es.indices.create(index="recipes", mappings=mappings)

#Index the data
bulk_data = []
for i,row in df.iterrows():
    bulk_data.append(
        {
            "_index": "recipes",
            "_id": i,
            "_source": {        
                "index": row['index'],
                "title": row['title'],
                "ingredients": row["ingredients"],
                "directions": row["directions"],
                "link": row["link"],
                "source": row["source"],
                "NER": row["NER"],
                "cleaned_NER": row["cleaned_NER"]
            }
        }
    )
bulk(es, bulk_data)

es.indices.refresh(index="recipes")
es.cat.count(index="recipes", format = "json")
