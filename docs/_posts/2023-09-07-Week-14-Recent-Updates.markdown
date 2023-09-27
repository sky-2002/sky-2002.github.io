---
title: '(GSoC) Week-14 Recent Updates'
categories: GSoC
tags: GSoC
---

Hello. This blog will be about a few updates I made in recent weeks. Also, I am excited to share that I will be giving a lightning talk in the **Google Summer of Code Virtual Contributor Lightning Talks 2023** on my project. Its going to be a short talk(3 minutes max), I will be summarizing my project and I am really excited for it!

In think blog, I will give some info on the recent updates, like adding coreference resolution, some small speedups we got in the code, etc.

#### **Background**
In the last blog, I had given details about how each component in our pipeline uses time to process each triple. Based on that, I tried to make things a little faster and we have achieved some speedup, earlier we took around `5 seconds` to process one-triple-containing-sentence(process means to do entity linking and relation mapping) and now we need around `1.8 seconds`.

Also, I have added co-reference resolution as discussed in the [TODOs blog]({% post_url 2023-08-06-Week-9-10-Mid-Eval-and-TODOs %}), this will help disambiguate entity mentions and thus we probably won't miss important relations due to entity not being recognized. Though, I haven't yet analysed the time consumption of this component.

#### **Speedup**
I found out that simply using transformers pipeline for GENRE instead of the model and tokenizer individually led to a `3.4 times` speedup in entity linking. The reasons for this speedup lies in the different optimizations that might be there in transformers pipeline.

#### **Next up?**
I will be looking to speed up and scale the pipeline further so as to run it on entire wikipedia.
