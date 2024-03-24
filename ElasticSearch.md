ElasticSearch
==============

    Port 9200 is used for all API calls over HTTP. This includes search and aggregations, monitoring and anything else that uses a HTTP request. All client libraries will use this port to talk to Elasticsearch
    Port 9300 is a custom binary protocol used for communications between nodes in a cluster. For things like cluster updates, master elections, nodes joining/leaving, shard allocation

 docker run --rm -p 9200:9200 -p 9300:9300 -e "xpack.security.enabled=false" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.12.2

 