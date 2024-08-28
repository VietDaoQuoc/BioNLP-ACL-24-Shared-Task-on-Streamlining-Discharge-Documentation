# BioNLP ACL'24 Shared Task on Streamlining Discharge Documentation
This is an university project that Viet Dao Quoc and Xuanzhi Chen took part in.
We deployed a RAG-model and used it together with different generative LLM-models for few-shot prompting for INFERENCE on the data.
In this project, our goal is to do model assessment with different models without FINE-TUNNING, the list of model in use:
- BioMistral-7B
- Standford's BioMedLM
- Qwen2-7B
- Meta-Llama3-70B
- Meta-Llama3.1-70B

Since we only had 2 months for doing this project we could only did the INFERENCE without FINE-TUNNING. 


# Deployment
We use Haystack for our Retrieval system with indexing.py is used to create the embedding for the documents. Whereas query.py is used with kNN-search to find top 3 best matches of a given query_input. 

Our setting include the followings over 3200 data samples from test_phase_2 dataset of the challenge using HuggingFace and Litellm with Anyscale-API:
- zero-shot prompting
- one-shot prompting with/without Retrieval Augmented Generation
- one-shot prompting with Retrieval Augmented Generation and with Multi-turn correction. (multi-turn means that we add another prompts to ask the model to correct its mistakes from its previous inference)

# Evaluation metrics
We used these following metrics for evaluation: 
- BERTScore
- ALIGNScore
- MEDCON
- BLEU
- ROUGE-L

# Result

As BioMistral-7B, BioMedLM, Qwen2-7B show
that they have the poor ability to follow instruc-
tions, we mainly measure Meta-Llama3-70B and
Meta-Llama-3.1-70B performance with the above-
mentioned metrics.
First, using the previously mentioned version of
the prompt, we assessed the Meta-Llama3-70B and
Meta-Llama3.1-70B models. We added instruc-
tions to remove the Discharge Medications part
for generating Discharge Instruction and Brief
Hospital Course across 3,200 samples. We take
the average of the scores from the evaluation for
each task, together with the overall score being an
average of all scores.
With Meta-Llama3-70B, we encountered some
scenarios where the total tokenized input-tokens
including the prompt exceeds max-input-tokens of
8192 tokens. We have tried to address the prob-
lem by reducing the total with the following ap-
proaches:
- Splitting query-input or sample-input into
smaller chunks, constructed smaller prompts
for those particular parts and then combine
the smaller outputs into one output. Limita-
tion: the run-time doubles for every prompt
for the model (also with Meta-Llama3.1-70B).
Therefore, we concluded this approach is not
realistic.
- Summarization of sample-input, we con-
structed another prompt to ask the model tocapture important informations of the sample-
input and then summarize it. The prompt for
this approach is zero-shot based. Litmitation:
therefore the output in this case contain lots
of hallucinations (up to 70 percent).

The above mentioned approaches were somewhat
promising but with our technical settings and
capability of the models, we decided just to filter
out instances that were causing the problem.
We further inspected that sample-input were not
the cause for exceeding max-input-tokens but
query-input. We decided using a filter that filters
out instances of query-input that together with the
prompt when tokenized has the length over 8192
tokens for version of One-shot with/without RAG.
Meanwhile there is another filter for One-shot
with RAG and with 1-Turn.
It is worth to notice that the run-time of the
Inference Pipeline for each query-input from
the test-phase-II dataset, using HuggingFace is
approximately around 89 second for both model.
The total run-time would be 734412 seconds which
is roughly 8.5 days. To speed up the process, we
reorganized our code and then used vLLM a is
a fast and easy-to-use library for LLM inference.
It then reduced the run-time to roughly less than
5 second for each single prompt, equivalent to 13
hours for the entire test-phase-II and 3 hours for
3200 samples of test-phase-II.

