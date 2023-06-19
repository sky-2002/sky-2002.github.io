---
title: '(GSoC) Week-2 Entity Linking - Intro'
categories: GSoC
tags: GSoC
---

Hello! In this blog, we will discuss about entity linking, a process used to map recognized entities to resources in a knowledge base, like DBpedia.

#### **Background**
Entity linking is the process of linking or mapping entity mentions(the identified entities in a text) to actual resources in a knowledge base. This is an important step towards our ultimate goal of relation extraction. Entity mentions in text need not be exactly same as the entities in the knowledge base, so it is very essential that we map them to the correct entity(in the KB) so as to extract correct relations. It is an important step which acts as a bridge to connect unstructured text to knowledge bases.

For example, consider the sentence "<span style="color:purple">England</span> <span style="color:orange">won the </span><span style="color:purple">2019 cricket world cup.</span>". So what does "<span style="color:purple">England</span>" refer to in this sentence, the country? Not exactly, it refers to the "<span style="color:purple">English cricket team</span>". Disambiguating such entity mentions can boost the relation extraction process. 

#### **Performing entity linking**

1. **DBpedia-spotlight** - DBpedia Spotlight is a tool for automatically annotating mentions of DBpedia resources in text, providing a solution for linking unstructured information sources to the Linked Open Data cloud through DBpedia.
{% gist d69a41b8e78d82e3a0547fa9600f5e8e %}

2. **DBpedia lookup** - The [DBpedia Lookup](https://github.com/dbpedia/dbpedia-lookup) service is an entity retrieval service for DBpedia entities that resolves keywords to resource identifiers. Thus it can be used to enrich plain text documents or tables with DBpedia URIs.
{% gist fe07e29f4c8b702e1fa44d7a38f99d40 %}

#### **Concepts**

- <u>Fuzzy matching</u> - It is different from exact matching in the sense that we have some tolerance for difference in the query and the retrieved results. Fuzzy matching is loose than exact matching, but very loose rules can lead more false positives. The system needs to account for misspellings, abbreviations, name variations, geographical spellings of certain names, shortened nicknames, acronyms, and many other variables.

- <u>Levenshtein Distance</u> - The Levenshtein distance is a number that tells you how different two strings are. The higher the number, the more different the two strings are. Below is a sample code to use this distance as well as some other distance options avaibale in the [rapid fuzz](https://github.com/maxbachmann/RapidFuzz) project.
{% highlight python%}
import rapidfuzz

s1 = "I am a student"
s2 = "I am studant"

print(f"Hamming distance - {rapidfuzz.distance.Hamming.distance(s1,s2)}") 
#=> 9
print(f"Levenshtein distance - {rapidfuzz.distance.Levenshtein.distance(s1,s2)}")
#=> 3
print(f"Jaro Winkler distance - {rapidfuzz.distance.JaroWinkler.distance(s1,s2)}")
#=>  0.062
{% endhighlight %}

I will keep this one a shorter blog in terms of concepts, but a more code and demo oriented. 

More about entity linking and the approaches I will use in the next blog!

Thanks!
{% include utterances.html %}