---
title: '(GSoC) Week-11, 12 and 13 Recent Updates'
categories: GSoC
tags: GSoC
---

Hello. Its been a while since I have put an update on the project(travelling and network issues). I had been working on some polishing, enhancing and re-experimenting, with the models that we are using, the time it takes to process the inputs etc. In this blog, I will give a review of these.

#### **Restating the Goal**
Just to make sure we are clear about the end goal, given a wikipedia page(an entity), we need to disambiguate the predicates/relationships between the page and all other pages(other entities) that are linked from that page. More precisely, we want to find the relationships between those pages which are only connected by the `dbo:wikiPageWikiLink` predicate and no other predicate, or mine the text to see even a relationship exists or not. There are cases where some page is linked in references, but is not related to the wikipedia page. The goal is to find all those hidden relationships by processing the wiki page text.

#### **Processing time for major components of pipeline**
**<u>Fetching wiki page and getting pure text</u>** - On an average, it takes about `0.76 seconds` to fetch a wikipedia page and extract pure text from it. Though this can be done in parallel for multiple pages. 

| Method | 1 Page | 4 Pages | 20 Pages |
|--------|--------|---------|----------|
| Sequential | 0.76 seconds | 3.04 seconds | 11.09 seconds |
| Parallel   | 0.76 seconds | 0.91 seconds | 3.05 seconds |

I tried experimenting using different number of processes to fetch 32 wiki pages, and below is the graph of the time taken in each case. (Time is in seconds)
<img src="/assets/images/Process_vs_fetch_1.png" alt= "Page fetching time vs processes" >

<!-- **<u>Splitting text into sentences</u>** - This task takes around `0.02 seconds` for each page on an average. -->

**<u>Extracting entities and relations with REBEL</u>** - This is one of the most important parts of the pipeline. With a GPU enabled, following are the times takes by the model for extraction.

| Number of sentences(sequential) | Time taken |
|---------------------------------|------------|
| 10 | 6.1 seconds |
| 100 | 53.18 seconds |

On an average, it takes around `0.5-0.6 seconds` per sentence. 

**<u>Entity Linking using GENRE</u>** - This step links the entity mentions extracted by REBEL with DBpedia entities. On an average, GENRE model takes around `2.26 seconds` to link one entity mention. This is probably the largest time consumer in the pipeline.

**<u>Relation mapping</u>** - This is the part where we map the textual relation with a DBpedia predicate based on the vector similarity score. This part works pretty fast with about `0.025 seconds` per relation(so we can say per sentence).

**<u>End-2-End Extraction</u>** - This time this takes depends on the number of triples that are extracted from the sentence that is passed to this. If we consider around 5 triples per sentence(considering long sentences), it takes around `25 seconds` on an average. This is justifiable as each triples has 2 entities and 1 predicate, so as per previous points, REBEL takes around `0.5 seconds`, then we need around  `2.26 x 2 = 4.52 seconds` for entity linking. So, it adds up to about **`5 seconds` per triple**. 

To understand the time taken for this end-2-end process, we need to get some insights into the lengths of the wikipedia texts, sentences etc. Below is a plot of the distribution of sentence lengths in terms of words:
<img src="/assets/images/sentence_length_dist_words.png">
The mean length of a sentence is about `26 words`. There is a high chance that such sentences contain more triples. And thus, more time to process.

#### **Some new issues with REBEL**
In one of the previous blogs, I had mentioned that REBEL goes wrong in case of factually correct sentences, but containing a negation. I found another issue with this model extracting wrong information.
For example, consider the image below:
<img src="/assets/images/rebel_issue_gh.png" alt= "REBEL Issue" width="800" height="300">

In the above example, the sentence is about `Bruce Springsteen's speaking in the middle of East Berlin` and not about `Berlin Wall being in the middle of East Berlin`. This is some unexpected behaviour. 

#### **Validation issues**
In the project description given [here](https://forum.dbpedia.org/t/towards-a-neural-extraction-framework-gsoc-2023/2083/4) and as mentioned while restating the goal, our aim is to disambiguate the relation between entities that are linked from a wiki page and thus only connected via `dbo:wikiPageWikiLink` predicate.. When we run the pipeline on a wiki page text, it gives a list of triples. These triples don't necessarily contain the current page/entity as the subject. So, in terms of validation, we can either remove those and only select the ones that have the current page as the subject entity, this will definitely solve our purpose. 

But when doing this, we may miss upon useful information regarding other entities. We need some way to keep these other triples stored or streamed somewhere so that we can still make use of them. 

One way to deal with this is that we store all triples and store them in a way such that we can get all triples with a specific subject entity together, so that we can disambiguate the relationships with that entity. This is similar to the map-reduce paradigm, something like this - the mapper emits relational triples, we accumulate them my the subject as the key, and the the reducer does the disambiguation task(checking if `dbo:wikiPageWikiLink` exists and no other predicate exists) for all triples for a particular subject. 

{% include utterances.html %}