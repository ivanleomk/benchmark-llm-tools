# Introduction

This is a repository showing a comparison between three different LLM Op providers. We compared them in two main ways

1. In production ( using a server ) and trying to log metrics such as average cosine similarity between query and retrieved results
2. Using Evals to test something simple like FTS vs Embedding Search

## Instructions

> You will need a .env file with the following variables
>
> ```
> LANGFUSE_SECRET_KEY=
> LANGFUSE_PUBLIC_KEY=
> LANGFUSE_HOST=
> ```
>
> In order to run the following code as well as an OPENAI_API_KEY shell variable

1. Install the required dependencies

```
uv venv
uv pip install -r requirements.txt
```

2. Generate the test dataset

```
python3 setup.py
```

2. We're using a simple fts recall and mrr prediction experiment here

```
python3 eval.py
```

3. To see the individual providers, we want to see

```
python3 server-<provider>.py
```

and we can make the same query to the `/query` endpoint

General Thoughts

- Langfuse is very slow for some reason and there isn't a way to easily compare across different metrics ( Benchmarked on ~ 900+ queries and it took 10 minutes to upload everything not including query execution time )
- Braintrust feels more mature as a solution ( ~50s including execution of queries )

- Braintrust is better than langfuse -> Easier to get started but requires some debugging ( But they need all scores to be between 0 -> 1)
- Langfuse ( Project to separate individual traces ( API key is per-project) ) while Brainrust has the concept of Projects ( u specify this in the init logger func)
- Traceloop : Concept of versioning in workflows but i suppose we can replicate that with metadata in Langfuse and Braintrust
