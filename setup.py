from datasets import load_dataset, Dataset
from lancedb import connect
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
import json
import os

func = get_registry().get("openai").create(name="text-embedding-3-small")


class DatabaseChunk(LanceModel):
    chunk: str = func.SourceField()
    embedding: Vector(func.ndims()) = func.VectorField()


def get_dataset_rows(dataset: Dataset, batch_size: int):
    curr = []
    for row in dataset:
        for passage in row["passages"]["passage_text"]:
            curr.append(
                {
                    "chunk": passage,
                }
            )

            if len(curr) == batch_size:
                yield curr
                curr = []

    if curr:
        yield curr


def get_dataset_queries(dataset: Dataset):
    for row in dataset:
        if not sum(row["passages"]["is_selected"]):
            continue

        yield {
            "query": row["query"],
            "passages": [
                row["passages"]["passage_text"][idx]
                for idx, is_selected in enumerate(row["passages"]["is_selected"])
                if is_selected
            ],
        }


if __name__ == "__main__":
    table_name = "ms-marco"
    n_rows = 200
    batch_size = 400
    db_path = "./db"
    os.makedirs("data", exist_ok=True)

    source_dataset = load_dataset(
        "ms_marco", "v1.1", split="train", streaming=True
    ).take(n_rows)
    # Create LanceDB Table
    db = connect(db_path)

    if table_name not in db.table_names():
        table = db.create_table(table_name, schema=DatabaseChunk)

        # Insert into LanceDB Table
        for passage_batch in get_dataset_rows(source_dataset, batch_size):
            table.add(passage_batch)

    with open("data/queries.jsonl", "w") as f:
        for query in get_dataset_queries(source_dataset):
            json.dump(query, f)
            f.write("\n")
