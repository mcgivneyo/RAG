"""
run with  python -m pytest test/test_prompts.py 
"""

import logging
import pytest
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics import context_recall, context_precision
from datasets import Dataset
from datasets import load_dataset
from src import rag

logger = logging.getLogger()

dataset = load_dataset('zeitgeist-ai/financial-rag-nvidia-sec')
n_eval = 5
queries = dataset['train'][:n_eval]['question']
gt_answers = dataset['train'][:n_eval]['answer']

# Test data
valid_queries = [(q,a) for q,a in zip(queries, gt_answers)]

hallucination_queries = [("hallucinated query", "This should not happen")]  # TODO

dont_know_queries = [
    ("What is the difference between a `creator", "I don't know"),
    ("What is tallest mountain in Europe ?", "I don't know"),
    ("Who won the grand national in 2020", "I don't know"),
]

dont_know_responses = [
    "I don't know",
    "I am not sure",
    "I cannot provide an answer",
    "I have no information on that",
    "I don't have the answer",
    "I am unable to answer that",
    "I don't have enough information",
    "I can't help with that",
    "I don't have the details",
    "I don't have an answer for that",
    "is not provided",
]


def test_prompt_1():
    """Validate that the output contains the expected phrases."""
    prompt = "What are some of the recent applications of GPU-powered deep learning as mentioned by NVIDIA?"
    output = rag.query_rag(prompt)
    print(output)
    assert all( [ phrase in output[0]["answer"] for phrase in [ "recommendation systems", "large language models", "generative AI" ]] )


@pytest.mark.parametrize("query,expected_answer", valid_queries)
def test_context_chunks_not_empty(query, expected_answer):
    """validate that the context chunks are not empty"""
    result = rag.query_rag(query)
    assert (
        len(result[0]["contexts"]) > 0
    ), f"Contexts should not be empty for query: {query}"


@pytest.mark.parametrize("query,expected_answer", valid_queries)
def test_chunks_are_strings(query, expected_answer):
    """validate that the chunks are strings"""
    result = rag.query_rag(query)
    assert all(
        isinstance(chunk, str) for chunk in result[0]["contexts"]
    ), f"Chunks should be strings for query: {query}"


@pytest.mark.parametrize("query,expected_answer", valid_queries)
def test_retrieve_data_not_empty(query, expected_answer):
    """validate the answer is not empty"""
    result = rag.query_rag(query)
    assert result[0]["answer"] != "", f"Answer should not be empty for query: {query}"


@pytest.mark.parametrize("query,expected_answer", valid_queries)
def test_retrieve_data_is_string(query, expected_answer):
    """validate the answer is a string"""
    result = rag.query_rag(query)
    assert isinstance(
        result[0]["answer"], str
    ), f"Answer should be a string for query: {query}"


@pytest.mark.parametrize("query,expected_answer", dont_know_queries)
def test_retrieve_data_dont_know(query, expected_answer):
    """validate that the output contains don't know responses for out of scope queries"""
    result = rag.query_rag(query)
    print(result[0]["answer"])
    assert any(
        [phrase in result[0]["answer"] for phrase in dont_know_responses]
    ), f"Answer should be 'I don't know' for query: {query}"


@pytest.mark.parametrize("query,expected_answer", valid_queries)
def test_retrieve_data_valid_queries(query, expected_answer):
    """validate expected answers with ragas_metrics"""
    context_precision_threshold = 0.3
    context_recall_threshold = 0.3
    faithfulness_threshold = 0.3
    answer_relevancy_threshold = 0.3

    result = rag.query_rag(query)
    dataset = Dataset.from_dict(
        {
            "question": [result[0]["query"]],
            "answer": [result[0]["answer"]],
            "contexts": [result[0]["contexts"]],
            "ground_truth": [expected_answer],
        }
    )

    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    ragas_metrics_result = evaluate(dataset=dataset, metrics=metrics)
    logger.info(ragas_metrics_result)
    assert (
        ragas_metrics_result["context_precision"][0] > context_precision_threshold
    ), f"Precision too low for query: {query}"
    assert (
        ragas_metrics_result["context_recall"][0] > context_recall_threshold
    ), f"Recall too low for query: {query}"
    assert (
        ragas_metrics_result["faithfulness"][0] > faithfulness_threshold
    ), f"Faithfulness too low for query: {query}"
    assert (
        ragas_metrics_result["answer_relevancy"][0] > answer_relevancy_threshold
    ), f"Answer relevancy too low for query: {query}"


# # # validate that hallucinations are not present TODO
# # @pytest.mark.parametrize("query,unexpected_answer", hallucination_queries)
# # def test_retrieve_data_no_hallucinations(query, unexpected_answer):
#       # validate that the output does not contain known or common hallucinated answers
# #     result = rag.query_rag(query)
# #     assert not any([phrase in result[0]['answer'] for phrase in ['hallucinated', 'answer']])
