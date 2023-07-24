---
title: '(GSoC) Week-8 End-2-End Relation extraction'
categories: GSoC
tags: GSoC
---

Hello. It feels amazing to start this blog as we have been able to create an end-2-end system for relation extraction from text, the ultimate goal of this <b>Neural Extraction Framework</b> project.

Before we begin, check the sample below of how the extracted and mapped triples look like:

<img src="/assets/images/e2e_1.png" alt= "Sample 1" width="800" height="300">

#### **Expectations**
The expectation from an end-2-end relations extraction is to take as input a wikipedia article text and return a set of relational triples. These relational triples are in the form of `<subject_entity> , <predicate> , <object_entity>/<literal>`. Also, initially we want these triples to have the wikipedia page itself as the subject entity. For example, if the page is about `Berlin_Wall`, we would like to pick the triples where the subject entity is `Berlin_Wall`. This way, we have same objective as extracting from infobox, just that we also make use of the text to extract relations and thus enhance the process to mine more information about that entity/page.

#### **Our approach**
In the series of previous blogs, we have discussed what tools we are using to accomplish the various tasks involved in relation extraction i.e extracting entities, linking them, extracting relation and linking it as well. For NER, we have options like `spacy` or some pretrained models from huggingface. For entity linking, we had a few approaches but after evaluating them, we finalized on GENRE. For relation extraction, we used REBEL(which also extracts entities, fyi) and to map those, we used vector similarity.


{% include mermaid.html %}
<div class="mermaid">
graph TD
    wiki_page[Wikipedia Page] --Extract plain text--> pure_text[Pure text]

    pure_text-->rebel(REBEL)
    rebel--as text-->entities[Entities]
    rebel--as text-->relations[Relations]
    
    relations--get embedding-->vector_similarity(Vector similarity with label embeddings);
    vector_similarity-->predicate_uris[Predicate URIs]

    pure_text-->annotation_for_genre(Annotate entities in text)

    entities-->annotation_for_genre

    annotation_for_genre--annotated text-->genre[GENRE]

    genre-->entity_uris[Entity URIs]
    
    entity_uris-->triples[Triples]
    predicate_uris-->triples[Triples]
    triples--Validate-->final_triples[Final triples]

    ;
</div>
<!-- entities-.->entity_linking(Other approaches for EL)
entity_linking-.->dbpedia_lookup(DBpedia Lookup)
entity_linking-.->redis_database(Redis database)
dbpedia_lookup-.->ensemble(Ensemble)
redis_database-.->ensemble
ensemble-.->entity_uris -->

#### **Explanation**
Below are some points that can help understand the above flow-chart better:
- The annotation of text with start and end of entities is a requirement of the GENRE models, hence that step.
- We have used the predicate labels available in DBpedia to get embeddings of predicates by using embedding models. In the image below, `work` is the text label for the predicate `https://dbpedia.org/ontology/Work`, we then use vector similarity over the text relation extraction by REBEL to find the best predicate for it.

<img src="/assets/images/e2e_2.png" alt= "Sample of label" width="900" height="300">

- Some validation over the extracted triples is needed which is discussed below.

#### **What next?**
- Check if the extracted triple already exists in the knowledge base.
- Validate the entities for the domain and range of the predicate.
- Check if the object entities are linked to subject entities via `dbo:wikiPageWikiLink`.
- There is also some scope of checking on the vector similarity that we are using for predicate mapping, because it is related to the embedding models we use, some work on evaluating their performance can help use further.

Will come with a blog on how this progresses, shortly.
Thank you!

{% include utterances.html %}
