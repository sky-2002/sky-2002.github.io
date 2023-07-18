---
title: '(GSoC) Week-6 Relation Extraction - REBEL'
categories: GSoC
tags: GSoC
---

Hello. The previous blog was the last in the entity-linking part of the project. Now we will move onto relation extraction. In this blog, we will discuss about REBEL - a relation extraction model. Given a text, this model can find relational triples(head-relation-tail) in the form of natural language text. 

#### **Background**
REBEL is a text2text model trained by [BabelScape](https://babelscape.com/) by fine-tuning <i>BART</i> for translating a raw input sentence containing entities and implicit relations into a set of triplets that explicitly refer to those relations. It has been trained on more than 200 different relation types.
REBEL is a joint model, meaning that it extracts entities and relations simultaneously.

### **Why Seq2Seq approach?**
Early approaches to relation extraction treated it like a pipeline - extracting entities first and then classifying the relation. Early end-2-end approaches that used transformers used neural networks to classify all word pairs in a text, much like using embeddings of two entities to predict the relation between them. The problem with these approaches is that they assume a single relation between an entity-pair. An overhead of these methods is that they need to find all word pairs in the text, which makes them computationally expensive.

Benefits of Seq2Seq approach to RE are that we can have decoding mechanisms to output same entities multiple times and also can be conditioned to avoid incompatible relations. 

#### **Mechanism**
REBEL expresses triplets as a sequence of tokens. For RE, we want to express triplets as a sequence of tokens such that we can retrieve the original relations and minimize the number of tokens to be generated so as to make decoding more efficient. <i>&lt;triplet&gt;</i> is a token that marks start of a triple, which is followed the surface form of the head entity. Then we have the <i>&lt;subj&gt;</i> token, that marks the end of head entity and start of tail entity. Finally <i>&lt;obj&gt;</i> token marks the end of the tail entity and start of the surface form of the relation.

(Note: These tokens are for the decoded part, see figure below)

They do something called triplet linearization. Sort entities by order of appearance in text. We begin with the first head entity, and group all the relations that start with it, then move onto the second head entity. 

<img src="/assets/images/triplet-linear.png" alt= "Triplet linearization" width="800" height="200">

We can see in the image above how we take the first head entity - "This must be the place" - and put in appropriate tokens all its relations and tail entities. We then begin with a new triplet with a new head entity. This also makes decoding text length less as we only decode the head entity once. 

While doing the joint NER and RE, they use <i>&lt;PER&gt;</i> or <i>&lt;ORG&gt;</i> tokens for entities. 

Find the demo code below:

{% gist 6e28a5e445db8bc0f0b860cf5d79b581 %}

Thank you!
{% include utterances.html %}