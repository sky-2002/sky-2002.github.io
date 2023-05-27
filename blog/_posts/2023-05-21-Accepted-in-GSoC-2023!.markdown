Hello and welcome!

I am delighted to write my first blog post about my acceptance in GSoC 2023 with [DBpedia](https://www.dbpedia.org/). Throughout the summer, I will work on the [Neural  extraction framework](https://summerofcode.withgoogle.com/programs/2023/projects/cKuagkf8) project with DBpedia.

#### **Background**:
It all started last year when I came to know about GSoC. It was that time when I learned about open-source, and I liked the idea of anyone being able to contribute and change the codes of so many softwares we use every day. I found an organization and worked hard contributing and writing a proposal. Though I was not selected for GSoC, I added two new distance metrics to a vector database later that summer. Since then, I have kept looking for opportunities in open-source projects, especially those I use.

And here I am, writing about my acceptance in GSoC 2023 for the Neural Extraction framework project. I would thank my mentors for giving me helpful feedback while writing the proposal. 


#### **The Neural Extraction Framework**:
The Neural Extraction project is in its third iteration this year, and I will be working towards enhancing and improving upon the work done in previous iterations. This project aims at extracting information in the form of relational triples(subject -> predicate -> object) from unstructured text in Wikipedia articles that can be added to the DBpedia knowledge base. 

<!-- ![Text has much more information](/assets/images/wiki_info_text.jpeg){:class="img-responsive"} -->
<img src="/assets/images/wiki_info_text.jpeg" alt= "Text has much more information" width="400" height="400">

Currently, tables and infoboxes are used to find relations between Wikipedia articles. In this project, we want to make use of the text of the article and identify wikilinks from entities in the wikipedia article, map those entities to resources in DBpedia, find the relation between the entities and then map this relation to a DBpedia property. 

<!-- For example, consider the sentence - "
<span style="color:red">
Messi
</span>
<span style="color:green">
is the captain of 
</span>
<span style="color:red">
Argentina
</span>"
, We need to extract the triples as follows:
![messi_captain_of_argentina](/assets/images/messi_captainof_argentina.png) -->

Throughout the proposal-writing process, I got a chance to read a lot of research papers related to NER, coreference resolution, relation extraction and also got a glimpse of their source code. Also, it was fun to try and test different approaches using popular frameworks like spacy, transformers etc. 

#### **Community Bonding period**:
After the contributors were announced, the GSoC team had organized "welcome talks" and some fun activities like the "GSoC contributor summit". Attending these was fun and also these were helpful to clear some doubts regarding the process. Our organization DBPedia also organized a meetup with all the contributors this year and it was amazing to meet all of them and learn about their projects.

As the coding period begins next week, I spent community bonding period learning about DBpedia, setting up myself for the coding period, exploring some amazing open source tools and also creating this blog site, this is the first time I am writing a blog!

As I will move through the project, I will try to document my learnings and project status through this. 

Thank you!