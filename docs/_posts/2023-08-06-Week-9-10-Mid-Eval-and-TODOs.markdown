---
title: '(GSoC) Week-9 and 10 Mid Evaluations and remaining TODOs'
categories: GSoC
tags: GSoC
---

Hello. I recently passed my mid term evaluation for GSoC 2023. It has been a great experience so far. DBpedia finally has an end-2-end neural extraction framework. Got insightful feedback from the mentors. In this blog, I will discuss about some tasks that still need to be done in order to make this end-2-end relation extraction reliable and correct.

#### **Background**
To give a quick overview - our end-2-end framework takes in raw text as input and gives a list of ```(<subject>, <predicate>, <object>)``` triples as output. We are using REBEL relation extraction model for joint entity-relation extraction, GENRE for entity linking and text based vector similarity for mapping relations to predicates. This pipeline is explained in the [previous blog]({% post_url 2023-07-24-Week-8-End-2-End-RE %}) and works great.

However there are some issues that need to be addressed in order for the framework to extract correct, useful and non-redundant triples. Below are the TODOs.

#### **TODOs**
Following are the tasks that need to be addressed.

- <u>Detecting literals in the object based on range</u> - In the current scenario, once REBEL returns us with the subject, relation and object from a sentence, we perform entity linking on the subject and object, that means we try to find an entity in DBpedia which is referred by the subject or object. But in some cases, the object can be a literal, like a date (for example ```1st January```). Instead of linking this to the DBpedia entity - [January 1](https://dbpedia.org/page/January_1), we want it to be stored as it is, a literal. Literals include dates, numbers, time durations, temparature readings, etc. We can probably use mappings based literals for this.


- <u>Check that the triple does not already exist</u> - This sounds trivial but can lead to redundant triples, as DBpedia knowledge graph is already populated. We need some way to check if the triple already exists, because it may happen that some relation is present in the infobox and thus in DBpedia, can also be extracted by our system from the text. 

- <u>Check that <span style="color:orange"><b>dbo:wikiPageWikiLink</b></span> exists</u> - As we want our system to extract all relational triples for a given wikipedia page, we also expect that these triples are related to the entity the page is about and other entities that are linked from this page. In order to satisfy this condition, we need to check for each triple that the subject entity is the same as referred by the wiki page and it is linked to the object entity by also the [<span style="color:orange"><b>dbo:wikiPageWikiLink</b></span>](https://dbpedia.org/ontology/wikiPageWikiLink) property. 

- <u>Evaluate embedding quality for relation mapping</u> - We need some way to compare different embedding models when we use them for relation mapping, i.e when linking a relation to a DBpedia predicate. I found these sentence embedding models to be particularly vulnerable to negations in sentences. For example:

    ```python
    # Consider the sentences
    p1 = "I used to play badminton professionally"
    p2 = "I had played badminton professionally"
    n1 = "I never used to play badminton professionally"

    encoder_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    jina_encoder_model = SentenceTransformer("jinaai/jina-embedding-t-en-v1")

    # Comparing similarity of (p1 and p2) and (p1 and n1) with both models

    # With MiniLM-L6-v2
    Similarity of two positive sentences p1 and p2: 0.9503945708274841
    Similarity of one positive p1 on negative n1: 0.762386679649353

    # With jina-embedding-t-en-v1
    Similarity of two positive sentences p1 and p2: 0.9439464211463928
    Similarity of one positive p1 on negative n1: 0.9112891554832458
    ```
    In the sentences above, we know that the similarity of the two positive (same meaning) sentences should be high whereas the similarity of the first positive sentence and its negation should be low, because ideally the two embeddings should be opposite to each other(if they really capture the meaning). In the two models we used, we can see that the MiniLM-L6-v2 is better at this task. Or maybe we can say that its threshold can be set around 85-90% for considering similar. Whereas for the Jina AI model, deciding the threshold seems difficult.

- <u>Validate predicate against properties - domain & range</u> - Curretly, when we extract the triples, we don't check if they are correct, or if they make sense. For example, it can happen that the entities do not fall in the domain or range of the predicate. We can check this by checking each entity after EL and discard the triple if it does not satisfy.

- <u>Handling relations with negation</u> - I was experimenting with the REBEL model, trying to understand how it responds to different input types etc and found out that dealing with sentences factually correct but containing negation, REBEL extracts wrong type of relation, ignoring the negation. For example, I passed in the following sentences to REBEL:
    ```python
    negative_samples = [
    "Messi never played for China",
    "Messi was not a member of Portugal Football team",
    "Obama is not a resident of India",
    ]

    # And the relations extracted were
    [{'head': 'Messi', 'type': 'member of sports team', 'tail': 'China'}]
    [{'head': 'Messi', 'type': 'member of sports team', 'tail': 'Portugal Football team'}]
    [{'head': 'Obama', 'type': 'residence', 'tail': 'India'}]
    ```
    The above sentences are factually correct, but contain negations. We can clearly see that REBEL extracts wrong(opposite) relations in them.

    For the discussion with the author, refer here - [Problem with negations in sentences](https://github.com/Babelscape/rebel/issues/66)

- <u>Add coreference resolution to get more relations</u> - This is because there may be sentences that may use pronouns to refer to an entity and have a relation. For example <span style="color:green">Messi is an Argentinian footballer. He previously played for PSG. Now he has joined Inter Miami</span>. In this sentence, <span style="color:green">he</span> refers to <span style="color:green">Messi</span>. So we would like the text to be coreference-resolved in order to not miss the relations in the second sentence.

We are working on these TODOs and would get back with some solutions in the next blog!

Thank you!
{% include utterances.html %}