# Nitro document understanding model
## Introduction
The Nitro project aims to enhance document comprehension using AI. This proof-of-concept (POC) explores transforming text-based documentation into data for training AI models. The experiment focuses on finetuning these models to improve their understanding of documentation.

## Overview
In this report, there are 2 experiments:
1. **Finetuning the Nitro Model:** Enhancing AI's ability to comprehend and assist with proprietary data.
2. **Model Comparisons:** Examining the differences between Finetuned model + RAG and Base model + RAG.

## 1. Finetuning the Nitro Model
With the growing need for AI assistants in various roles, like customer service, we explore the potential of Large Language Models (LLMs) for this purpose. This section details our approach to finetuning an LLM on documentation to assess its accuracy in answering questions.

### Data generation
This study focuses on text-based data, specifically Nitro's markdown-formatted documentation. Given LLMs' context size limitations, we break the documentation into smaller, manageable chunks. The aim is to reformat unstructured text into a structured, trainable dataset using the `instruction` finetuning approach. This method, effective in various specialized models, involves creating question-and-answer pairs or instruction-response pairs to guide the model. There are a few successful models using this technique, namely [Math specialized model](https://arxiv.org/abs/2308.09583), [Code specialized model](https://arxiv.org/abs/2306.08568), and [Search specialized model](https://github.com/SciPhi-AI/agent-search). Take them as a reference, we also follow that approach for `Nitro documentation specialized model`.

To generate this dataset, we utilize `GPT-4` for its efficiency and proven success in data generation, as demonstrated in several specific models. Our process includes:

- Chunking the text into 300-token chunks with a 30-token overlap to maintain coherence. This is to ensure after generating QnA pairs, we don't exceed the maximum context length of the LLM (4096 tokens).
- Creating a `system prompt` for GPT-4 to generate QnA pairs, optimizing API call efficiency by generating multiple pairs per call. In our study, we use a method inspired by the [Evol instruct](https://github.com/nlpxucan/WizardLM) approach to create questions of varying complexity. However, we've simplified this process to make it easier to implement.
- Postprocessing the data for upload to the Huggingface platform.

The dataset is available on [HuggingFace](https://huggingface.co/datasets/jan-hq/nitro_binarized_v2).

### Finetuning
We adopt Supervised Finetuning (SFT) using Huggingface's Zephyr Beta guidelines in [alignment-handbook](https://github.com/huggingface/alignment-handbook). We also explore optimal hyperparameters in `LoRA`, like varying `R` (rank) and `Alpha` (scaling factor), which significantly impact the model's trainable parameters (e.g. smaller R and Alpha lead to small trainable parameters). Our training configurations are available in `config_lora.yaml`.

### Results
We generated approximately `3800 diverse QnA pairs` from the Nitro documentation.

In this experiment we chose [Stealth v1.2 7B](https://huggingface.co/jan-hq/stealth-v1.2) as the base model, are as follows:

Table 1. Training result of Nitro models
| Model       | r | alpha   | Loss  | Time|
|-------------|-------|-----|-------|---|
| Nitro V1 E1 | 16    | 32  | 1.185 |3m|
| Nitro V1 E3 | 16    | 32  | 0.853 |10m|
| Nitro V1.2 E1 Qlora | 256  | 512 | 0.6513|6m|
| Nitro V1.2 E3 Qlora | 256  | 512 | 0.3123|18m|

*E: epochs, m: minutes*
**Note:** Nitro V1.1 is omitted as it was an initial test with 2000 samples.

From the table, we can see that even with [QLoRA](https://arxiv.org/abs/2305.14314), the technique that trains in 4bits with large `R` and `Alpha` hyperparameters, we can still reduce loss after training.

We conducted practical tests by asking the model various questions through [Jan](https://jan.ai/). Here's what we found:

- **Understanding of Nitro Documentation:** The model successfully learned new information from Nitro's documentation, as shown in `Image 1`. This indicates that it accurately understands details about Nitro.

![Alt text](img/nitro_know.png)

Image 1. Accurate Knowledge of Nitro by the Model

- **General Responses:** In some cases, however, the model provided generic answers instead of Nitro-specific information, as seen in Image 2. This suggests a need for further refinement in context-specific responses.

![Alt text](img/nitro_install.png)

Image 2. General Response from the Model

- **Inaccuracies in Responses:** There were instances where the model generated incorrect information, highlighting areas for improvement in accuracy.

![Alt text](img/nitro_windows.png)

Image 3: Incorrect Information Provided by the Model

### Limitation
- Potential evaluator biases in the current evaluation.
- Exclusive use of QLoRA for finetuning may affect performance.

### Key takeaway
In this experiment, we've learned:
- GPT-4 generated QnA pairs are effective for transforming unstructured data into a trainable format.
- SFT with large `R` and `Alpha` values significantly enhances the model's comprehension of documentation.

### Future work
- **Develop a Benchmark for Model Evaluation:** We plan to create a benchmark to assess the performance of our fine-tuned model. This will help us understand its strengths and areas for improvement more clearly.
- **Enhance Data Generation Methods:** These will include technical questions about code and general knowledge questions. This approach will make our training data more diverse and detailed, helping our model to learn and perform better.
- **Create a DPO Dataset:** This dataset will be instrumental in further improving the model's accuracy and deep understanding of new data.

## 2. Comparative Analysis: Finetuned model + RAG and Base model + RAG
In this section, we compare the performance of Finetuned and Base models using Retrieval Augmented Generation (RAG) to understand documentation. We hypothesize that a finetuned model, combined with RAG, will outperform a base model in answering documentation-related questions.

### Experiment Setup
- **RAG System Simplification:** We constructed a basic RAG (Retrieval-Augmented Generation) system using the [Llamaindex](https://www.llamaindex.ai/) `version 0.9.33` preset for straightforward implementation. For the embedding, we chose [bge-en-base-v1.5 embedding](https://huggingface.co/BAAI/bge-base-en-v1.5), which offers a good balance between size and performance. We configured the `chat template` to ChatML to fit our Large Language Model (LLM) to ensure we get the most out of it. The documentation was split down into chunks of 300 tokens, with overlaps of 30 tokens. To keep things simple, we didn’t include a `ranker model` in our RAG system. Moreover, we used the same `system prompt` for every model in our experiment.

```
You are a helpful and careful assistant. You will use the given context to answer the multiple choice question. Only response 1 letter (A, B, C or D).
```

Details of our RAG system can be found in [nitro_rag.ipynb](foundry/experiments/stealth-on-nitro/rag/nitro_rag.ipynb).

- **Models**: For the experiment, we choose 4 subjects to test out
    1. **Base model**: The base model is [Stealth v1.2 7B](https://huggingface.co/jan-hq/stealth-v1.2) trained on general knowledge but doesn't have specific information about Nitro documentation.
    2. **Base model + RAG:** This is the combination of a base model with a RAG system.
    3. **Finetuned model**: This is a best-performing Nitro-specific model from our previous experiments [Nitro V1.2 E3 Qlora](https://huggingface.co/jan-hq/nitro-v1.2-e3)
    4. **Finetuned model + RAG:** This is the combination of a finetuned model with a RAG system.

- **Initial Testing:** We created 50 open-ended questions covering various aspects of Nitro documentation, ranging from simple facts to complex, interconnected queries. We did 3 rounds of testing, each round had 50 questions to see the responses of 4 models. The question list is available [here](foundry/experiments/stealth-on-nitro/rag/mcq_nitro.csv).

- **Benchmarking:** To avoid bias, we transformed these questions into a multiple-choice format (MCQ) with four options (A, B, C, D), using answers from the initial tests and refining them for accuracy. The dataset also includes deliberately incorrect or misleading options. The revised dataset is accessible [here](foundry/experiments/stealth-on-nitro/rag/mcq_nitro.csv).

- **Evaluation Plan:** We conducted 200 rounds of testing on 4 models, each with 50 questions, to assess the hypothesis:
    - Null Hypothesis (H0): Fine-tuning the RAG model on the set of documents does not significantly improve the performance or effectiveness of information retrieval and generation compared to using the RAG model without fine-tuning.
    - Alternative Hypothesis (H1): Fine-tuning the RAG model on the set of documents significantly improves the performance or effectiveness of information retrieval and generation compared to using the RAG model without fine-tuning.
    - Statistical significance was determined with a p-value of 0.05, using t-tests.

## Result

**NOTE: This report includes statistics from 132 completed rounds.**

- **Initial Testing:** 
In Table 2, The initial rounds of testing showed promising results, indicating that the `finetuned model combined with RAG` is better than the `base model with RAG`.
    - **Base model:** Stealth v1.2 7B lacked knowledge about Nitro, resulting in 0% accuracy.
    - **Fine-Tuned Model:** The finetuned model alone achieves 55% accuracy, indicating an improvement in understanding Nitro documentation compared to the base model.
    - **Base + RAG:** The combination of RAG and the base model elevates the accuracy to 62%, suggesting that RAG compensates for the base model's lack of domain-specific training.
    - **Fine-Tuned + RAG:** The highest accuracy is observed here at 79%, showcasing that combining finetuning and RAG significantly enhances the model's ability to understand and respond accurately to Nitro documentation-related queries.

Table 2. Initial result
| Model        | Percentage |
|--------------|------------|
| Base         | 0%         |
| Fine-Tuned   | 55%        |
| Base + RAG          | 62%        |
| Fine-Tuned + RAG | 79%  |

Following these results, the `Base only model` was excluded from further evaluation to streamline the evaluation process.

- **Descriptive statistics**
Unfortunately, we have to exclude the `Finetuned only` model from the test because, with the same prompt, it can't produce a proper answer (A, B, C, D) to MCQs.

In Table 3, we provide a more detailed look into the performance of the `Fine-Tuned Model + RAG` and the `Base Model + RAG` across multiple rounds of testing.
    - **Mean Accuracy:** The `Fine-Tuned Model + RAG` mean accuracy is higher (57.8%) than the `Base Model + RAG` (47.9%), indicating a consistently better performance.
    - **Standard Deviation:** A lower standard deviation in the `Fine-Tuned Model + RAG` (1.81%) versus the `Base Model` (2.15%) implies that the finetuned model's performance is more consistent.

Table 3. Descriptive table

|           | Fine-Tuned Model + RAG | Base Model + RAG | ChatGPT + RAG | ChatGPT4 + RAG |
|-----------|----------------------|------------|---------|----------|
| count     | 200.000000           | 200.00000  | 168.000000 | 89.000000 |
| mean      | 0.576400             | 0.47720    | 0.567262 | 0.643371 |
| std       | 0.019415             | 0.02242    | 0.011669 | 0.021582 |
| min       | 0.520000             | 0.44000    | 0.540000 | 0.580000 |
| 25%       | 0.560000             | 0.46000    | 0.560000 | 0.640000 |
| 50%       | 0.580000             | 0.48000    | 0.560000 | 0.640000 |
| 75%       | 0.580000             | 0.50000    | 0.580000 | 0.660000 |
| max       | 0.640000             | 0.54000    | 0.600000 | 0.680000 |


- **T-test and p-value**
We use an independent t-test to test the statistical significance of the observed differences. The `p-value` here is extremely low (< 0.0001), suggesting that the observed differences in performance are statistically significant and not due to random chance. Therefore, we reject the null hypothesis and conclude that the alternative hypothesis is correct. This suggests that fine-tuning a model on the set of documents improves its performance in information retrieval and generation compared to using the RAG system without a fine-tuned model.

Table 4. Hypothesis testing statistic 
| Statistic      | Value                     |
|----------------|---------------------------|
| Test Statistic | 40.387         |
| p-value        | 1.632e-114   |

- **Probability Distribution of Model Accuracies**
    - **Distribution Spread:** The `finetuned model + RAG` distribution is more peaked and less spread out, indicating higher performance consistency, as mentioned. In contrast, the `base model + RAG` distribution is wider and flatter, implying more variability in its performance.
    - **Overlap of Distributions:** There is some overlap between the two distributions, suggesting that there are instances where the `base model + RAG` might perform as well as the `finetuned model + RAG`. However, such instances are less frequent, as the relatively small overlap area indicates.
    - **Outliers or Anomalies:** The `finetuned model + RAG` distribution shows a small bump around the accuracy of `0.62`, which might indicate the presence of outliers where the model performed significantly better than its average.
    - **Hypothesis Testing Implications:** The clear separation between the means of the two distributions and the low p-value from the hypothesis testing support the alternative hypothesis (H1) that finetuning the model on the set of documents improves the performance.

![Alt text](img/nitro_distribution.png)

Image 2. Probability Distribution of Model Accuracies

- **Investigate which is the most wrong questions**
For further insight, we conducted an analysis to identify which types of questions each model struggled with. The following table (Table 4) lists the top 10 questions where the `Base Model + RAG` performed poorly, but the `Finetuned Model + RAG` excelled.

Table 4. Top 10 questions
| Question                                      | Finetuned Model + RAG | Base Model + RAG |
|----------------------------------------------|----------------------|------------|
| Teach me quickstart                           | 1.0                  | 0.00       |
| How to use embedding?                        | 1.0                  | 0.00       |
| How can I contribute to Nitro?               | 1.0                  | 0.00       |
| Show me steps to build from source on Windows | 1.0                  | 0.00       |
| I want to enable multi-thread                | 1.0                  | 0.00       |
| What is the benefit of using Nitro?          | 1.0                  | 0.00       |
| I want to download a model; how can I do it? | 1.0                  | 0.00       |
| How to do model health check in Nitro?        | 1.0                  | 0.00       |
| How to use the continuous batching feature?   | 1.0                  | 0.00       |
| How do I make an inference?                  | 1.0                  | 0.05       |

Unfortunately, we can't exactly point out whether this is the fault because of the RAG system or the LLM itself.

## Limitations
- Lack of detailed logs to determine if inaccuracies are due to the RAG system or the LLM.
- Use of a basic RAG system, which may not reflect optimal performance.
- We can't test the `Finetuned model without RAG system`.

## Key take away
- By finetuning the model with documentation, the `finetuned model + RAG` performance is better than the `base model + RAG` by nearly 10%.

## Future work
- Improve the RAG system with better logging and applying advanced RAG techniques.
- Further finetuning the model by using RAG information as well as Nitro documentation.
