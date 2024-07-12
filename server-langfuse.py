from fastapi import FastAPI
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv
from pydantic import BaseModel
import instructor
from langfuse.openai import openai
import lancedb
from langfuse import Langfuse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
app = FastAPI()


langfuse = Langfuse()

client = instructor.from_openai(openai.OpenAI())

db = lancedb.connect("./db")
table = db.open_table("ms-marco")


class Response(BaseModel):
    chain_of_thought: str
    response: str


@observe()
def embed_query(query: str):
    return (
        openai.embeddings.create(
            input=query,
            model="text-embedding-3-small",
        )
        .data[0]
        .embedding
    )


@observe()
def retrieve_context(query: str, query_embedding: list[float]):
    langfuse_context.update_current_trace(
        tags=["context", "retrieval"],
        user_id="1234",
        session_id="2345",
        metadata={
            "version": "v0.1.0",
            "model": "ms-marco",
            "current_system": "ms-marco",
            "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Golden_Gate_Bridge_as_seen_from_Battery_East.jpg/500px-Golden_Gate_Bridge_as_seen_from_Battery_East.jpg",
        },
    )
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

    langfuse_context.score_current_observation(
        name="Cosine Similarity",
        value=avg_cos_sim,
        comment=f"Average Cosine Similarity retrieved for query - {query}",
    )

    langfuse_context.update_current_observation(
        metadata={"observation": "this is a test about the overall metadata"}
    )

    return retrieved_context


@observe()
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


@app.post("/query")
@observe()
def query(query: str):
    query_embedding = embed_query(query)
    context = retrieve_context(query, query_embedding)
    response = generate_response(query, context)
    return response
