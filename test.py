from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics import context_recall, context_precision
from datasets import Dataset
from src import rag

prompt = "What are the benefits of an event-driven architecture?"
output = rag.query_rag(prompt)

data = {
        "question": [output[0]["query"]],
        "answer": [output[0]["answer"]],
        "contexts": [output[0]["contexts"]],
        "ground_truth": ['tbd'],
    }
dataset = Dataset.from_dict(data)
metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
result = evaluate(dataset=dataset, metrics=metrics)
print(result)
