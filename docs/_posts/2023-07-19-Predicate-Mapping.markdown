---
title: '(GSoC) Week-7 Predicate mapping'
categories: GSoC
tags: GSoC
---

Hello. In this blog, we will be looking into mapping natural language relations to predicates in DBpedia. This task is extremely important in order to achieve our ultimate goal of a neural extraction framework.

#### **Background**
In the previous post, we had discussed about REBEL model for relation extraction. This model extracts the head entity, relation and tail entity from texts. But they are in natural language form - for example, "lives in" - we want them to be in the form of URIs - for example "https://dbpedia.org/ontology/residence" - so that we can link them to dbpedia predicates. For mapping entities to dbpedia resource URIs, we already have a few methods for entity linking. 


#### **Method used**
Simply put, we are using vector/embedding similarity to map natural language relations to predicates in DBpedia. Now the question is, how do we create vectors for text and these URIs. For text, we have a lot of options, and we have used `sentence-transformers` for these. To map each URI to a vector, we made use of the labels that are associated with these predicate URIs.

Based on the label for each predicate, we pass in that label(which is in natural language) through the `sentence-transformer` model to get an embedding, and we associate this embedding with the predicate and use it for vector similarity.

So, when we have a textual relation, we get its embedding, and we compare it with the embeddings of all predicates that we have and select that predicate which has the highest score. See sample code below.
```python
from sentence_transformers import SentenceTransformer
encoder_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_sentence_transformer_model(model_name):
    model = SentenceTransformer(model_name_or_path=model_name)
    return model

def get_embeddings(labels, sent_tran_model):
    embeddings = sent_tran_model.encode(labels, show_progress_bar=False)
    return embeddings
```
The model we have used for encoding the labels is based on BERT with 6 Transformer Encoder Layers,
it can handle 512 tokens and return dense vector representation with 384 features.

To perform the search, we use a `gensim` word2vec model and load it with the stored label embeddings that we have pre-computed using `sentence-transformers`. We then return that predicate whose label embedding is most similar to the embedding of the query, in this case, the natural language relation extracted by rebel.


#### **Any alternative methods?**
For the method described above, we need to collect the predicates, their URIs and the labels and then use it. We can use something like a paraphrase dictionary, like [this one](https://github.com/pkumod/Paraphrase/tree/master), and use the relations and predicates from there.

Thank you!

{% include utterances.html %}