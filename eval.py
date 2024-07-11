from lancedb import connect
from lancedb.table import Table
import json
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from dotenv import load_dotenv
from langfuse import Langfuse
import itertools
from tqdm import tqdm
from braintrust import Eval, Score

load_dotenv()
langfuse = Langfuse()

func = get_registry().get("openai").create(name="text-embedding-3-small")


class DatabaseChunk(LanceModel):
    chunk: str = func.SourceField()
    embedding: Vector(func.ndims()) = func.VectorField()


def calculate_mrr(predictions: list[str], gt: list[str]):
    mrr = 0
    for label in gt:
        if label in predictions:
            mrr = max(mrr, 1 / (predictions.index(label) + 1))
    return mrr


def get_recall(predictions: list[str], gt: list[str]):
    return len([label for label in gt if label in predictions]) / len(gt)


eval_metrics = [["mrr", calculate_mrr], ["recall", get_recall]]
sizes = [3, 5, 10, 15, 25]

metrics = {
    f"{metric_name}@{size}": lambda predictions, gt, m=metric_fn, s=size: (
        lambda p, g: m(p[:s], g)
    )(predictions, gt)
    for (metric_name, metric_fn), size in itertools.product(eval_metrics, sizes)
}


def get_queries(path: str):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def fts_search(queries: list[str], table: Table, k: int):
    return [
        [
            row["chunk"]
            for row in table.search(query, query_type="fts")
            .select(["chunk"])
            .limit(k)
            .to_list()
        ]
        for query in tqdm(queries)
    ]


def get_query_questions(queries: list[str]):
    return [query_obj["query"] for query_obj in queries]


def get_query_passages(queries: list[str]):
    return [query_obj["passages"] for query_obj in queries]


def score(predictions: list[list[str]], labels: list[list[str]], metrics: list[str]):
    return [
        {metric: score_fn(prediction, label) for metric, score_fn in metrics.items()}
        for prediction, label in zip(predictions, labels)
    ]


def evaluate_braintrust(input, output, **kwargs):
    return [
        Score(
            name=metric,
            score=score_fn(output, kwargs["expected"]),
            metadata={"query": input, "result": output, **kwargs["metadata"]},
        )
        for metric, score_fn in metrics.items()
    ]


def log_with_langfuse(
    scores: list[dict], queries: list[str], results, metadata: dict[str, str], trace_tag
):
    for eval_score, result, query in tqdm(zip(scores, results, queries)):
        trace = langfuse.trace(
            name=trace_tag,
            input=query,
            metadata={"query": query, **metadata},
            output=result,
        )

        for metric in eval_score:
            trace.score(name=metric, value=eval_score[metric])


if __name__ == "__main__":
    providers = ["LANGFUSE", "BRAINTRUST"]

    db = connect("./db")
    table = db.open_table("ms-marco")
    table.create_fts_index("chunk", replace=True)

    query_objs = get_queries("data/queries.jsonl")

    queries = get_query_questions(query_objs)
    labels = get_query_passages(query_objs)

    if "LANGFUSE" in providers:
        import time

        start = time.time()

        fts_results = fts_search(queries, table, 25)
        fts_scores = score(fts_results, labels, metrics)

        log_with_langfuse(
            fts_scores, queries, fts_results, {"search_type": "fts"}, "fts-search"
        )
        langfuse.flush()

        print(f"Took {time.time() - start}s")

    if "BRAINTRUST" in providers:
        import time

        start = time.time()
        Eval(
            "MS Marco",  # Replace with your project name
            data=lambda: [
                {
                    "input": query,
                    "expected": label,
                    "metadata": {"search_type": "fts", "k": "25"},
                }
                for query, label in zip(queries, labels)
            ],  # Replace with your eval dataset
            task=lambda query: [
                row["chunk"]
                for row in table.search(query, query_type="fts")
                .select(["chunk"])
                .limit(25)
                .to_list()
            ],
            scores=[evaluate_braintrust],
        )
        print(f"Took {time.time() - start}s")
        # Eval(
        #     "MS Marco",  # Replace with your project name
        #     experiment_name="fts",
        #     data=lambda: [
        #         {
        #             "input": query,
        #             "expected": label,
        #             "metadata": {"search_type": "hybrid", "k": "25"},
        #         }
        #         for query, label in zip(queries, labels)
        #     ],  # Replace with your eval dataset
        #     task=lambda query: [
        #         row["chunk"]
        #         for row in table.search(query, query_type="hybrid")
        #         .select(["chunk"])
        #         .limit(25)
        #         .to_list()
        #     ],
        #     scores=[evaluate_braintrust],
        # )
