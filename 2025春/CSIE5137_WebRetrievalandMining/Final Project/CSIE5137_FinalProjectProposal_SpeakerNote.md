# Speaker Notes for Cross-Lingual Fact-Check Retrieval Presentation

## Title Slide
Good afternoon everyone. Today we're presenting our project proposal on "Cross-Lingual Fact-Check Retrieval Using Contrastive Learning and Knowledge Distillation." This is our final project for the Web Retrieval and Mining course. We are Yi-Xuan Jiang, James, and Ting-Kuan Hsieh, and we'll be walking you through our approach to tackle this challenging problem.

## Outline
Here's an outline of our presentation. We'll start with an introduction to the problem, then discuss our proposed methodology, planned experiments, and our project timeline. This will give you a comprehensive overview of what we aim to accomplish.

## Problem Statement
The core challenge we're addressing is the rapid spread of misinformation across language barriers on social media platforms. We're participating in SemEval-2025 Task 7, which focuses on multilingual and crosslingual fact-check retrieval. Our goal is to develop a system that can effectively retrieve relevant fact-checked claims for social media posts across multiple languages. This would allow fact-checkers to quickly determine if a claim has already been fact-checked, even when the original claim and fact-check exist in different languages.

## Why This Matters
This problem is worth solving for several important reasons. First, fact-checking organizations have limited resources, and automation can help prioritize truly novel claims. Second, misinformation frequently crosses language boundaries, and tracking these patterns is crucial. Third, speed is essential when combating viral misinformation, and automated retrieval can significantly reduce response time. Fourth, many fact-checking organizations are limited to specific languages, and a crosslingual system expands their effective coverage. Finally, by improving fact-checking processes, we can help reduce the harmful effects of misinformation on society.

## SemEval 2025 Task 7 Overview
Let me provide more context about the specific shared task we're addressing. SemEval-2025 Task 7 focuses on retrieving relevant fact-checked claims from a large multilingual database when given a social media post. The task includes multiple languages like English, Spanish, French, German, Italian, Portuguese, Arabic, Hindi, and others. The primary challenge is finding semantic equivalence across languages. We'll be working with the MultiClaim dataset, which contains posts and fact-checks in multiple languages. Success will be measured primarily using success@10, which evaluates whether the correct fact-check appears in the top 10 retrieved results.

## Multilingual Dense Retrieval Architecture
Our core approach involves a dual-encoder architecture based on pre-trained multilingual language models. This architecture will encode both social media posts and fact-checked claims into a shared embedding space, where similarity can be efficiently computed. We'll explore base models like XLM-RoBERTa, mBERT, and LaBSE, which have shown strong performance in multilingual tasks.

## Key Technical Approaches
We're implementing three key technical approaches. First, contrastive learning with in-batch negatives and hard negative mining to train the model to distinguish between relevant and irrelevant fact-checks. Second, knowledge distillation, where we'll leverage larger teacher models like multilingual-T5 or BLOOM to distill knowledge into more efficient student models. And third, translation-augmented training to specifically address crosslingual challenges by augmenting our training data with translations.

## Retrieval Enhancement Techniques
To further enhance retrieval performance, we'll implement several techniques. For query expansion, we'll use multilingual resources like WordNet or BabelNet to add synonyms and related terms. We'll also employ a two-stage retrieval approach with lightweight initial retrieval followed by more sophisticated cross-encoder reranking. For multimodal integration, we'll extract and utilize text from images using OCR. And finally, we'll implement language-specific processing with robust language identification and customized tokenization and normalization for different language families.

## Generative Pseudo Labeling
We're particularly excited to incorporate techniques from Chen et al.'s 2024 EACL paper on "Unsupervised Multilingual Dense Retrieval via Generative Pseudo Labeling." This approach addresses the crucial challenge of limited parallel data in low-resource languages by using large language models to generate pseudo-aligned text pairs. This will help us bridge gaps between languages with limited resources and improve cross-lingual retrieval performance.

## Anticipated Challenges
We anticipate several challenges in this project. Semantic drift across languages is a major concern, as concepts may not map perfectly between languages. Computational efficiency will be crucial, requiring a balance between performance and practical inference speed. Data imbalance is likely, with some languages having significantly more training examples than others. Cultural context is also important, as misinformation often relies on cultural references that may not translate well. Finally, translation quality in the dataset may vary and impact model performance.

## Evaluation Metrics
For evaluation, we'll primarily use success@10 as specified by the SemEval task, which measures whether the correct fact-check appears in the top 10 retrieved results. For our internal development, we'll also track additional metrics like Mean Reciprocal Rank, Normalized Discounted Cumulative Gain, and Precision and Recall at various cutoffs. We'll conduct language-specific performance analysis to identify potential strengths and weaknesses across different language pairs.

## Ablation Studies
To understand the contribution of each component, we'll conduct thorough ablation studies. We'll compare performance with and without OCR text, evaluate different embedding models, measure the impact of knowledge distillation, assess the value of translation augmentation, and analyze the effect of generative pseudo labeling. We'll also conduct language-specific analysis to understand performance across different language pairs and the impact of language families on crosslingual retrieval.

## Datasets and Resources
Our primary dataset will be the MultiClaim dataset provided by the SemEval-2025 Task 7 organizers. We'll supplement this with additional resources like the CLEF CheckThat! Lab datasets, the MultiLingual Misinformation Dataset, and data from fact-checking websites. Our technical stack will include Hugging Face Transformers for model implementation, PyTorch as our deep learning framework, and FAISS for efficient similarity search.

## Experimental Setup
For our experimental setup, we'll implement a rigorous cross-validation approach with 5 folds while preserving language distribution. We'll use 4 folds for training and 1 for validation. Hyperparameter tuning will be done using Bayesian optimization, focusing on embedding size, learning rate, and batch size. We'll conduct language-specific performance evaluation and detailed error analysis by claim type and language pair to identify areas for improvement.

## Project Timeline
Here's our project timeline aligned with the course deadlines. We've just submitted our proposal on May 2nd. From May 3rd to 8th, we'll be exploring the data and setting up our project repository. On May 9th, we'll review feedback and adjust our plans accordingly. From May 10th to 16th, we'll implement baseline models. From May 17th to 23rd, we'll focus on developing our core multilingual embedding models. From May 24th to 30th, we'll implement advanced features like contrastive learning and knowledge distillation. From May 31st to June 4th, we'll integrate and test our system, and finally, on June 6th, we'll deliver our presentation and submit our final report.

## References
Here are the key references for our project, including the paper we've added on Unsupervised Multilingual Dense Retrieval via Generative Pseudo Labeling by Chen et al. These works provide the foundation for our approach and will guide our implementation.

## Thank You
Thank you for your attention. We're excited about this project and its potential impact on multilingual fact-checking workflows. We welcome any questions or suggestions you might have.