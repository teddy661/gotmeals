from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd
from pathlib import Path 

#USING THIS FOR FRONT END PART

def load_data_and_index(es, csv_file_path):
    # Load the data from the CSV file
    if not Path(csv_file_path).exists():
        raise FileNotFoundError("CSV file not found")

    df = (pd.read_csv(csv_file_path, engine='python')
          .dropna()
          .sample(5000, random_state=42)
          .reset_index()
          )

    # Define the mappings for the index
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

    # Create the index with the specified mappings
    if es.indices.exists(index="recipes"):
        es.indices.delete(index="recipes")  # Delete the existing index if it exists
    es.indices.create(index="recipes", mappings=mappings)

    # Index the data into Elasticsearch
    bulk_data = []
    for i, row in df.iterrows():
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

    # Refresh the index to make the changes visible
    es.indices.refresh(index="recipes")
    return es.cat.count(index="recipes", format="json")
