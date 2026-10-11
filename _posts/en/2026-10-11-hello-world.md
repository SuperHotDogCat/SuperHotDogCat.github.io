---
title: "Starting My Blog"
categories: [Diary]
tags: [Jekyll, GitHub Pages]
description: "I built a portfolio site with Jekyll and GitHub Pages."
---

I built a portfolio-slash-blog with Jekyll and GitHub Pages.

## Why I made it
- It started to feel like not having a proper portfolio site was getting a bit embarrassing.
- I wanted to write up my master's research not only on arXiv but also on a personal blog.
- Until now I'd been posting on [Note](https://note.com/cute_orchid39) and [Qiita](https://qiita.com/SuperHotDogCat), but blogging platforms these days are flooded with AI-generated content, so I decided to go back to a personal blog. Everything on this site is definitely checked by a human.
  - I do use AI to help with writing, but my policy is that any results or code are actually run and verified by me (as long as my motivation hasn't run out).
- As of 2026 (who knows whether it'll still be used in the future), there isn't a single article, in Japan or elsewhere, that properly explains Unified Memory over NVLink-C2C, which is also slated for NVIDIA Vera Rubin, and I felt I had to get something written down.
- In particular, by some twist of fate I ended up working with [CUDA Extended GPU Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html) through both my undergrad and master's, so I believe I have a thing or two to say about it.

## Choosing the tech
- This portfolio site is built with Jekyll.
- There was a time when I was stuck on the idea that a portfolio site ought to be built with a modern framework like Next.js, got fed up, and never made one. But thinking about it calmly, my portfolio site needs no server-side rendering, no database, and no backend. What matters more is that my motivation doesn't evaporate whenever I go to update it, so I went with Jekyll, which my former lab also uses. Now I can update the site just by writing Markdown. Thanks to the senior members of my lab!
- The design is based on the site of [Lei Mao](https://leimao.github.io/), which helped me enormously back when I was learning CUDA. Thank you! I've never met Lei Mao in person or interacted with them at all, but it's no exaggeration to say that the experience of writing a GEMM from scratch that matches cuBLAS performance shaped my current research, my career path, and my engineering skills in high-performance computing.
- People who love programming contests like AtCoder probably only care about big-O notation for algorithms like matrix multiplication, but in real HPC, the hardware you use, memory bandwidth and the memory hierarchy, and even things like page size matter more than computational complexity. Speeding up GEMM on a GPU is a great subject for learning all of that.
