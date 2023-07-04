---
title: '(GSoC) Week-5 Entity Linking - Benchmarking'
categories: GSoC
tags: GSoC
---

Hello. In this blog, I will discuss about the benchmarking(or evaluation 😅) of different entity-linking approaches that I have discussed in previous few blogs. I have used the WNED dataset for benchmarking/evaluation.

#### **Dataset**
I used the WNED(WNED-WIKI from the KILT Benchmarking) dataset available [here](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.7939/DVN/10968). The dataset contains 345 wikipedia articles and 6,821 entity mentions from them, along with the entities they refer to(as wikipedia article titles). In the dataset, there are raw-text wikipedia article content which contain entity mentions, and then there is an `wikipedia.xml` file that contains information like the article title, each mention and its associated entity. I collected all entity mentions and their entities in a file:
<img src="/assets/images/sample-data-el.png" alt= "Sample entity mentions" width="500" height="150">

#### **Benchmarking / Evaluation**
For this, I applied 3 methods - using DBpedia lookup, using a redis-database of entities and surface forms, and the GENRE model. 
- **DBpedia lookup** - I passed each entity mention to lookup and it returned a dbpedia resource.
- **Redis-database** - Similar to lookup, I passed in an entity mention and it returned a dbpedia resource.

A caveat of this approach is that we don't make use of the context, in different contexts, same entity mention can point to different entities. For example, "<span style="color:purple">England</span> <span style="color:orange">won the </span><span style="color:purple">2019 cricket world cup.</span>". "<span style="color:purple">England</span>" here, refers to the "<span style="color:purple">English cricket team</span>", not the country. But if we pass <span style="color:purple">England</span> to the lookup or the redis database, they return the England country as entity, which is not what we want.

- **GENRE** - As this is an autoregressive model, and the way it is trained and made available for inference, we can pass to it a text with annotations for entity-mention, which we want to link to some entity. This model then autoregressively generates this entity. I had discussed about the process by which it does this generation in the [previous blog]({% post_url 2023-06-27-Week-4-Entity-Linking %}). A great advantage of this model is that we can pass in the context(in my case, the sentence containing the entity mention) to the model, based on which it generates the entity. Thus it associates the mention with a more relevant entity.

Some examples:

1. We can see how lookup and redis-db both failed at correctly linking the entity. I have used only the sentence containing the entity mention as the context, but we can alter it and make it include more text.

    ```
    Mention: Wales
    Entity: Wales national rugby union team
    URI: http://dbpedia.org/resource/Wales_national_rugby_union_team

    Lookup: http://dbpedia.org/resource/Wales
    Redis: http://dbpedia.org/resource/Wales
    GENRE: Wales national rugby union team
    Sentence: He then played against Scotland on 26 February and [START_ENT] Wales [END_ENT]  on 12 March.
    ```

2. The example below shows how GENRE could connect `Ga` with elements and correctly link it to `Gallium`.

    ```
    Mention: Ga
    Entity: Gallium
    URI: http://dbpedia.org/resource/Gallium

    Lookup: http://dbpedia.org/resource/Atlanta
    Redis: http://dbpedia.org/resource/Ga-Adangbe_people
    GENRE: Gallium
    Sentence: The elements in question are the  [START_ENT] Ga [END_ENT] , Ge, As, Se and Br.
    ```

3. GENRE correctly disambiguated by making sense out of the `lacrosse` tournament.

    ```
    Mention: Princeton
    Entity: Princeton Tigers men's lacrosse
    URI: http://dbpedia.org/resource/Princeton_Tigers_men's_lacrosse

    Lookup: http://dbpedia.org/resource/Princeton_University
    Redis: http://dbpedia.org/resource/Princeton_University
    GENRE: Princeton Tigers men's lacrosse
    Sentence: In 1881, Harvard defeated [START_ENT] Princeton [END_ENT]  to win the first intercollegiate lacrosse tournament.
    ```

#### **Results**

| Approach | Accuracy (exact match) |
|----------|------------------------|
| DBpedia lookup | 43.92% |
| Redis-database | 61.82% |
| GENRE | 78.80% |

#### **Interesting stats**
- Though GENRE had the highest accuracy, it was not correct in predicting for 5% of the entity mentions.
- For almost 20% entity mentions, only GENRE was correct in predicting the associated entity.
- For 2.5% of the entity-mentions, only redis-database gave correct entities.
- For around 16% entity mentions, none of the methods could link to correct entity. This means at least one of them was correct for almost 84% of the entity mentions.
- Around 40% of the times, all methods were correct.

Thank you!

{% include utterances.html %}