from fastapi import FastAPI
from dotenv import load_dotenv
from braintrust import init_logger, traced, wrap_openai, current_span
from pydantic import BaseModel
import instructor
import lancedb
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
app = FastAPI()

logger = init_logger(project="Braintrust Logging")


db = lancedb.connect("./db")
table = db.open_table("ms-marco")


class Response(BaseModel):
    chain_of_thought: str
    response: str


oai = wrap_openai(openai.OpenAI())
client = instructor.patch(oai)


@traced
def embed_query(query: str):
    return (
        oai.embeddings.create(input=query, model="text-embedding-3-small")
        .data[0]
        .embedding
    )


@traced
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


@traced
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
    current_span().log_feedback(
        scores={"cosine_similarity": avg_cos_sim, "cos_2": avg_cos_sim}
    )
    current_span().log(
        # input={"query": query, "query_embeddings": query_embedding},
        # output=retrieved_context,
        metadata={
            "version": "v0.1.0",
            "model": "ms-marco",
            "current_system": "ms-marco",
            "k": 25,
            "image": {
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Golden_Gate_Bridge_as_seen_from_Battery_East.jpg/500px-Golden_Gate_Bridge_as_seen_from_Battery_East.jpg",
            },
        },
    )
    return retrieved_context


@app.post("/query")
@traced
def query(query: str):
    query_embedding = embed_query(query)
    retrieved_context = retrieve_context(query, query_embedding)
    response = generate_response(query, retrieved_context)
    current_span().log(metadata={"test": "123"})
    return response
