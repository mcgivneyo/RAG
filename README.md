
# Basic Rag with Evaluation

* a simple rag implementation using llama3.2 running locally on ollama
* for fast development and iteration can also use gpt-4o-mini (requires access key)
* Switching between models can be selected using a parameter in the llm  completion function (RAGAs metrics always uses chatgpt)  


# todo
log in to hugging face hub
huggingface-cli login



# download nvidia 2023 SEC Filing Report
# !wget https://stocklight.com/stocks/us/nasdaq-nvda/nvidia/annual-reports/nasdaq-nvda-2023-10K-23668751.pdf


update the env




## Running the app

Add a .env file with openai key
```
OPENAI_API_KEY=sk...
```

Start the local LLM:  
```
ollama run llama3.2:latest
```
Run the main script:
```
python3.11 ./src/rag.py 
```




## 1. Adding tests to validate the model output.  

__Metrics:__  
I used the sample questions to create synthetic ground truth answers using gpt-4o. These are a proxy for human labelled gold data set answers. 
The sample questions are noisy and require futher refinement to remove irrelevent, or context-free questions not specific to the scraped text eg "what is the name of the new platform?"
that make them unsuitable for RAG evaluation. Additional work should be done to add filtering/scoring for "don't know" answers

To evaluate the retrieval part: 
I used hit-rate and Mean Reciprocal Rank to score whether the chunks for each document were retrieved from the vector database for the corresponding question.
1.  Hit rate: scores 1 if any of the chunks from the document were returned from the vector database (ie did the retrieve identify any of the chunks corresponding to the question)
2.  Mean Reciprocal Rank: measures the average of the inverse of the rank of the first retrieved chunk (ie was the retrieve able to find the most relevant chunks for each question)
3.  Precision: measures the proportion of retrieved chunks that belong to the corresponding question

To evaluate the generation part: 
I used out-of-the-box ragas metrics: These are "llm as a judge" based metrics, calculated using chatgpt 

1. context_recall: Context Recall measures how many of the relevant contexts (or pieces of information) were successfully retrieved.
2. context_precision: measures the proportion of relevant chunks in the retrieved contexts
3. Faithfulness: Faithfulness metric measures the factual consistency of the generated answer against the given context. It is calculated from answer and retrieved context. The answer is scaled to (0,1) range. Higher the better. The generated answer is regarded as faithful if all the claims made in the answer can be inferred from the given context. 
4. Relevancy: assesses how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information and higher scores indicate better relevancy. 

 My view is that metrics like these can be useful to guide optimisation and fine-tuning, assuming that human labelled data in unavailable. None of these metrics is perfect, the results and scoring are non-deterministic so the absolute scores may not in themselves be meaningful. Also manual inspection and review may also required to validate the metrics and failure cases. Evaluation and scoring of RAG pipelines, and llm hallucination detection in general, is an area of ongoing development.

### 3.  HOw to improve the chunking for the RAG?
Given the short length and lack of structure and formatting in the scraped text I created a simple character-length parser with an overlap parameter. A recursive character parser could be used to create better chunks, that are split at paragraph, sentence, word level etc. The character length and overlap length parameters are hyper-parameters to be tuned, in combination with the number of retrieved chunks.


## How else would you improve this RAG system? 

__Retrieval Improvements:__
* fine tune chunking parameters
* simple heuristics to add more context to each chunk: eg pre-pend title
* use a better embedding model
* use semantic chunking
* add a re-ranking model
* use an ensemble chunk-size strategy (ie multiple embeddings of the same document at different chunk sizes) combined with a document retrieval heuristic
* experiment with combinations of different embedded text vs retrieved text, small-to-big, sentence window etc
* summarise each document and add to the corpus, update the document retrieval heuristic
* depending on llm context window size and average document size, retrieve the entire document based on identified chunks
* use hybrid vector search
* add guard-rails and query modification methods to refine and improve the queries


__Generation Improvements:__
 - use a better generation model
 - optimise the prompt for llm, text, query type
 - add (few shot) examples to the prompt for a particular style and format
 - fine tune an llm (e.g. with LoRA) on sample answers
 - for use-cases beyond basic factoid retrieval investigate REACT and Chain-of-thought prompts that iterate over the generated text and request the LLM to evaluate its own output





