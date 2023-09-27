---
title: '(GSoC) Summary of Neural Extraction Framework'
categories: GSoC
tags: GSoC
---

Hello. This is the last blog related to GSoC as my project nears to an end and the final evaluation approaches. In this blog, I will try to summarize the project, write about what could be added further and future scope etc. 

#### **Summary of Neural Extraction Framework**

- GSoC project page - [Link](https://summerofcode.withgoogle.com/programs/2023/projects/cKuagkf8)

- Github Repo - [Link](https://github.com/dbpedia/neural-extraction-framework/tree/main/GSoC23)

[Neural Extraction Framework](https://github.com/dbpedia/neural-extraction-framework/tree/main/GSoC23) has been a wonderful experience for me. This project was in its third iteration this year in the Google Summer of Code and I am delighted to share that we have achieved to have an end-2-end framework to automatically extract relational triples from wikipedia articles and link them to entities and predicates in DBpedia.

Our goal was to extract relations between entities(wikipedia articles) which could not be extracted by just parsing the infoboxes. This needed us to extract not only the entities/mentions and link them to entities in Dbpedia but also the relations, along with linking the relations to predicates.

We used the [REBEL](https://github.com/Babelscape/rebel) model from Babelscape to jointly extract entity and relations(in textual form), then to link them to entities in DBpedia, we used [GENRE](https://github.com/facebookresearch/GENRE) from facebook research. After that, to link the relations to predicates, we represented each predicate in DBpedia with an embedding by applying sentence-transformer on the text label of those predicates and to the relation extracted by REBEL. At the end, we used vector similarity to link relations to predicates.

We designed this as a pipeline which takes in a link for a wikipedia page, extracts text from it, does some pre-processing, applies the REBEL model, then performs the entity linking and relation mapping and finally emits out triples extracted from that wikipedia article. This pipeline of ours takes about `1.8 seconds per triple`, which is quite slow(we don't shy away to point out our shortcomings). In this last 2-3 weeks, I have tried to explore some options to speed it up further, but couldn't apply those changes to our pipeline. 

#### **Shortcomings of the current system**
- Speed
- Dependency between components.
- Dependance on DL models (not exactly a shortcoming, but using something classical and simpler will be good).

#### **Possible extentions and future scope**
- Speed has been an issue for our pipeline. The DL models that we have used take time to process input and also because of the structure and type of input they need, we couldn't highly parallelize the pipeline other than at few places.
- We did plan and try to scale this pipeline to entire wikipedia. But we haven't been able to yet. Our plan was to use some distributed processing frameworks like Spark, [Ray](https://www.ray.io/) or [SparkNLP](https://sparknlp.org/).
- The current pipeline depends on two deep learning models(REBEL and GENRE) for relation extraction and linking. The relation extraction model that we use is a joint entity-relation system, we could think of these tasks as separate as well. For this, I have a separate module for NER and relation extraction in the codebase. As newer models for these tasks keep cmoing in, we could easily integrate them.
- Also, special focus to some type of entities or relations can be given, for example, we can say that we would design a system to only extract causal relationships. This gives us the freedom to use classical approaches as well(of course with latest enhancements) or a hybrid of DL and classical approaches.
- Parallelizing and making the most out of resources wherever possible. This time, our aim was to get an end-2-end framework, and that is the reason why speed, parallelism or scale were not the primary targets. But now that we have an end-2-end system, we can focus on speed, scaling etc. 
- A possible python framework that is handled and maintanied by the community. 


#### **My experience in GSoC 2023**
<img src="/assets/images/gsoc_selection_ss.png" alt= "My project selection screenshot">

Google summer of Code was an amazing experience for me. Right from the time of writing the application, getting it reviewed, updating it and finally waiting for the results. The result day was also exciting, I waited till midnight but couldn't see the selected contributors and didn't receive any email. But in the morning, I woke up to selection emails that made my day. I would keep looking at my GSoC dashboard and feel good about it, because I had tried last year as well, but couldn't make it, so this was special for me.

I had a great learning experience, especially in NLP domain. I came to know about autoregressive retrieval, generation models, embedding models etc. Along with this, I got to explore sentence-transformers, spacy, transformers, flair and other such packages. I became more fluent with Git as well. Also, maintaining these blogs for GSoC has been a good experience for me, I learnt about jekyll and github pages.

Finally, I would like to express my sincere gratitude towards my mentors - [Tommaso Tsoru](https://uk.linkedin.com/in/tommasosoru) and Ziwie XU - for being the best mentors I could get. They have been very helpful and I have got some really amazing insights from them throughout the project. This project would not have been possible without them.

I look forward to continue contributing to this project in future as well!

Thank you!!