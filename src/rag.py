"""basic rag implementation"""

import ast
import logging
from pathlib import Path
from typing import Union, List
from pymilvus import MilvusClient, utility, connections
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics import context_recall, context_precision

load_dotenv()
logging.basicConfig(level=logging.INFO)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

DB_URI = "./resources/milvus_db.db"
COLLECTION_NAME = "nvidia_collection"
client = MilvusClient(DB_URI)
DIMENSION = 384  # size of embedding vectors
TEXT_CUT_OFF = 500

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", 
    device="mps"
)


def split_files( doc_text:str, chunk_size=1200, chunk_overlap=100):
    """
    split document into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    doc_texts = text_splitter.create_documents([doc_text])
    return doc_texts


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_prompts(question: str, contexts: str) -> tuple[str,str]:
    """
    get rag prompts for the LLM model
    args:
        question: str: question
        contexts: str: contexts
    returns:
        str: system prompt
        str: user prompt
    """
    system_prompt = (
        "You are an expert at understanding and answering questions about the NVIDIA SEC filing"
    )
    user_prompt = (
        "Answer the question below using only the texts provided in Context. If you don't "
        "know the answer, say that you don't know. Use a maximum of two sentences and keep "
        "the answer concise. Only answer the question, do not add extra information. "
        f"\nQuestion: {question} \nContext: {contexts} \nAnswer:"
    )
    return system_prompt, user_prompt


def get_llm_completion(system_prompt: str, user_prompt: str, llm_model: str) -> str:
    """
    get llm completion from openai or llama running locally on ollama
    args:
        system_prompt: str: system prompt
        user_prompt: str: user prompt
        llm_model: str: name of llm : "gpt-4o-mini", "gpt-4o", "llama3.2"
        returns:
            str: completion
    """
    base_url = "http://localhost:11434/v1"
    if llm_model in ["gpt-4o-mini", "gpt-4o"]:
        llm_client = OpenAI()       # openai model
    elif llm_model == "llama3.2":   # llama 3.2 model running locally
        llm_client = OpenAI(base_url=base_url, api_key="ollama")
    else:
        logging.error("model %s not supported", llm_model)
        return "llm not supported"

    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        top_p=0.5,
        max_tokens=300
    )
    return response.choices[0].message.content


def get_embedding(text: list[str], show_progress_bar=True):
    """get text embeddings
    args:
        text: str: text to get embedding for
    returns:
            np.array: embedding
    """
    return embedding_model.encode(text, show_progress_bar=show_progress_bar)


def create_collection(
    chunks: list, collection_name: str = COLLECTION_NAME, dimension: int = DIMENSION
    ) -> None:
    """
    create collection in milvus
    args:
        chunks: list: list of document chunks
        collection_name: str: collection name
        dimension: int: dimension of the embedding
    returns:
        dict: response from milvus
    
    todo: update the metadata for subject etc
    """
    res = {}
    embeddings = embedding_model.encode(chunks)
    data = [{"id": id, "vector": embeds, "text": doc, "subject": ""}
            for id, (embeds, doc) in enumerate(zip(embeddings, chunks))]
    try:
        client.create_collection(collection_name=collection_name, dimension=dimension)
        res = client.insert(collection_name=collection_name, data=data)
        logging.info("inserted %d docs into vector db", res.get("insert_count", 0))
        return
    except ConnectionError as e:
        logging.error("Failed to connect to the database: %s", e)
    except ValueError as e:
        logging.error("Value error: %s", e)
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
        raise e


def clean_text(text: str) -> str:
    """clean text"""
    return text.lower().strip()


def retrieve_chunks(
    query: str,
    collection_name: str = COLLECTION_NAME,
    num_results: int = 5,
    verbose: bool = True,
    ) -> dict:
    """
    main rag query function
    args:
        query: str: query
        collection_name: str: collection name
        num_results: int: number of results
        verbose: bool: verbose
    returns:
        dict: response from milvus
    """
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    query = clean_text(query)
    query_embed = get_embedding([query], show_progress_bar=False)

    res = client.search(
        collection_name,
        query_embed,
        output_fields=["id", "text", "subject"],
        params=search_params,
        limit=num_results,
    )
    if verbose:
        for doc in res[0]:
            logging.info(
                "id %s distance %.3f %s",
                doc["id"],
                doc["distance"],
                doc["entity"]["text"][:TEXT_CUT_OFF],
            )
    return res


def delete_collection(collection: str = COLLECTION_NAME, db_uri: str = DB_URI) -> None:
    """
    delete milvus vector db collection
    args:
        collection: str: collection name
        db_uri: str: uri of db
    returns:
        None
    """
    connections.connect(db_name="default", uri=db_uri)
    utility.drop_collection(collection)


def init_collection(docname:str, n_chars=500, n_overlap_chars=100) -> None:
    """
    utility class to reset and create vector db
    args:
        None
    returns:
        dict: chunk_dict {chunk_id: doc_id}
    """
    pdf_text = get_pdf_text([docname])
    chunks = split_files(pdf_text)
    delete_collection()
    create_collection(chunks=[doc.page_content for doc in chunks])
    return


def query_rag(questions: Union[List[str], str], llm_model="gpt-4o-mini") -> list:
    """
    get rag responses for a list of queries
    args:
        queries: Union[List[str], str]: A list of questions or a single question.
        llm_model: str: llm model to use
    returns:
        list: list of responses with query id, query, chunk ids and answer
    """
    rag_responses = []
    questions = [questions] if isinstance(questions, str) else questions
  
    for query in questions:
        resp = retrieve_chunks(query=clean_text(query), verbose=False)
        contexts = [doc.get("entity", {}).get("text") for doc in resp[0]
                        if doc.get("entity", {}).get("text")]
        sys_prompt, u_prompt = get_prompts(query, "\n ".join(contexts))
        rag_response = get_llm_completion(
            system_prompt=sys_prompt, user_prompt=u_prompt, llm_model=llm_model
        )
        logging.info("query : %s", query)
        logging.info("answer: %s", rag_response)
        rag_responses.append(
            {
                "query": query,
                "contexts": contexts, 
                "answer": rag_response
            }
        )
    return rag_responses


def get_test_data(n_samples=25) -> tuple:
    """
    get test data and synthetic ground truth approximation for evaluation
    args:
        n_samples: int: number of samples
    returns:
        tuple: test_questions, test_doc_ids, ground_truth
    """
    data = pd.read_csv("resources/ee_case_studies.csv")
    data.sample_questions = data.sample_questions.map(ast.literal_eval)
    data["idx"] = data.index
    data["questions"] = data["sample_questions"].apply(lambda x: x.get("questions"))
    data = (
        data.explode("questions")
        .drop_duplicates(subset=["questions"])
        .reset_index(drop=True)
    )
    test_questions = data["questions"].tolist()
    test_doc_ids = data["idx"].tolist()

    # get ground truth data
    if Path("resources/ground_truth.csv").exists():
        ground_truth_df = pd.read_csv("resources/ground_truth.csv")
    else:
        logging.info("creating ground truth for evaluation")
        ground_truths = []
        for idx, query, text in data[["idx", "questions", "text"]].values:
            sys_prompt, u_prompt = get_prompts(query, text)
            rag_response = get_llm_completion(
                system_prompt=sys_prompt, user_prompt=u_prompt, llm_model="gpt-4o"
            )
            ground_truths.append((idx, query, rag_response))

        ground_truth_df = pd.DataFrame(
            ground_truths, columns=["index", "question", "ground_truth"]
        )
        ground_truth_df.to_csv("resources/ground_truth.csv", index=False)

    ground_truths = ground_truth_df.ground_truth.to_list()
    if n_samples:
        logging.info("sampling first %d questions out of %d", n_samples, len(data))
        test_questions = test_questions[:n_samples]
        test_doc_ids = test_doc_ids[:n_samples]
        ground_truths = ground_truths[:n_samples]

    return test_questions, test_doc_ids, ground_truths


def get_eval_metrics(eval_data: list, gt_answers: list) -> None:
    """
    create RAG evaluation metrics
    1. evaluate retrieval metrics using chunk_map
    2. evaluate generation metrics using ragas
    args:
        eval_data: list: list of evaluation data
      
    returns:
        None
    """
    # df_eval = pd.DataFrame(eval_data)
    # find the document that each chunk came from
    # df_eval["chunk_doc"] = df_eval.chunk_ids.map(
    #     lambda c_ids: [chunk_map.get(id) for id in c_ids if chunk_map.get(id)]
    # )
    # # hit rate: 1 if any returned chunks came from the target document
    # df_eval["hit_rate"] = df_eval[["doc_id", "chunk_doc"]].apply(
    #     lambda x: 1 if x.doc_id in x.chunk_doc else 0, axis=1
    # )
    # # mean reciprocal rank: 1 / (rank of first returned chunk from target document)
    # df_eval["mrr"] = df_eval[["doc_id", "chunk_doc"]].apply(
    #     lambda row: (
    #         1 / (row.chunk_doc.index(row.doc_id) + 1) if row.doc_id in row.chunk_doc else 0
    #     ),
    #     axis=1,
    # )
    # # precision: proportion of returned chunks that came from the target document
    # df_eval["precision"] = df_eval[["doc_id", "chunk_doc"]].apply(
    #     lambda row: (
    #         sum([cd == row.doc_id for cd in row.chunk_doc]) / len(row.chunk_doc)
    #         if row.chunk_doc
    #         else 0
    #     ),
    #     axis=1,
    # )

    # logging.info("Retrieval Evaluation results:")
    # logging.info("MRR: %f", df_eval.mrr.mean())
    # logging.info("Hit Rate: %f", df_eval.hit_rate.mean())
    # logging.info("Precision: %f", df_eval.precision.mean())

    data = {
        "question": [row["query"] for row in eval_data],
        "answer": [row["answer"] for row in eval_data],
        "contexts": [row["contexts"] for row in eval_data],
        "ground_truth": gt_answers,
    }
    dataset = Dataset.from_dict(data)
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    result = evaluate(dataset=dataset, metrics=metrics)
    df_res = result.to_pandas()
    df_res.to_csv("resources/rag_results.csv", index=False)

    logging.info("Generation Evaluation results:")
    logging.info("Context Precision: %f", df_res.context_precision.mean())
    logging.info("Context Recall: %f", df_res.context_recall.mean())
    logging.info("Faithfulness: %f", df_res.faithfulness.mean())
    logging.info("Answer Relevancy: %f", df_res.answer_relevancy.mean())
    return


if __name__ == "__main__":
    dataset = load_dataset('zeitgeist-ai/financial-rag-nvidia-sec')
    n_demo = 3
    n_eval = 5

    logging.info("creating document embeddings...")
    init_collection(docname="resources/nasdaq-nvda-2023-10K-23668751.pdf")

    logging.info("starting demo RAG test queries...")
    queries = dataset['train'][:n_demo]['question']
    _ = query_rag(queries)

    logging.info("starting RAG evaluation...")
    queries = dataset['train'][:n_eval]['question']
    gt_answers = dataset['train'][:n_eval]['answer']

    rag_data = query_rag(queries)
    get_eval_metrics(eval_data=rag_data, gt_answers=gt_answers)
    logging.info("RAG evaluation completed")
