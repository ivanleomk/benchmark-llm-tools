from traceloop.sdk import Traceloop
from dotenv import load_dotenv
from traceloop.sdk.decorators import workflow, task
from fastapi import FastAPI
import lancedb
import instructor
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel

load_dotenv()
app = FastAPI()

oai = openai.OpenAI()
client = instructor.from_openai(oai)

Traceloop.init()

db = lancedb.connect("./db")
table = db.open_table("ms-marco")


class Response(BaseModel):
    chain_of_thought: str
    response: str


def embed_query(query: str):
    return (
        oai.embeddings.create(input=query, model="text-embedding-3-small")
        .data[0]
        .embedding
    )


@task(name="Generate Response")
def generate_response(query: str, context: list[str]):
    formatted_context = "\n-".join(context)
    return client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are about to be passed a user query to respond. Make sure to cite the relevant passages in your response. Here is some relevant contex.\n{formatted_context}",
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        model="gpt-3.5-turbo",
        response_model=Response,
        max_retries=3,
        temperature=0.5,
    )


@task(name="Retrieved Context")
def retrieve_context(query: str, query_embedding: list[float]):
    retrieved_items = table.search(query).limit(25).to_list()

    retrieved_items_embeddings = np.array(
        [np.array(item["embedding"]) for item in retrieved_items]
    )

    retrieved_context = [item["chunk"] for item in retrieved_items]

    # Calculate cosine similarity between query_embedding and retrieved_items_embeddings
    avg_cos_sim = (
        cosine_similarity(
            np.array(query_embedding).reshape(1, -1), retrieved_items_embeddings
        )
        .flatten()
        .mean()
    )
    Traceloop.set_association_properties(
        {
            "cos_sim": avg_cos_sim,
            "version": "v0.1.0",
            "model": "ms-marco",
            "current_system": "ms-marco",
            "k": 25,
            "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Golden_Gate_Bridge_as_seen_from_Battery_East.jpg/500px-Golden_Gate_Bridge_as_seen_from_Battery_East.jpg",
        }
    )
    return retrieved_context


@app.post("/query")
@workflow(name="Generate User Query", version="1")
def generate_user_query(user_query: str):
    query_embedding = embed_query(user_query)
    retrieved_context = retrieve_context(user_query, query_embedding)
    response = generate_response(user_query, retrieved_context)
    Traceloop.set_association_properties({"test": "123"})
    return response
