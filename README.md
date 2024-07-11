# Introduction

This is a repository showing a comparison between three different LLM Op providers. We compared them in two main ways

1. In production ( using a server ) and trying to log metrics such as average cosine similarity between query and retrieved results
2. Using Evals to test something simple like FTS vs Embedding Search

## Instructions

1. Install the required dependencies

```
uv venv
uv pip install -r requirements.txt
```

2. Your lancedb DB is included with the git repo so you can immediately run a simple evaluation exercise using the different providers. We're using a simple fts recall and mrr prediction logging here.

General Thoughts

- Langfuse is very slow for some reason and there isn't a way to easily compare across different metrics ( Benchmarked on ~ 900+ queries and it took 10 minutes to upload everything not including query execution time )
- Braintrust feels more mature as a solution ( ~50s including execution of queries )

```
python3 eval.py
```
