# RAGTAG

A tool for manual RAG chunk entry for question/answer systems.  Create, search or edit text chunks paired 
up with questions to ensure good retrieval for embeddings.

Takes advantage of Atlas Mongo and Atlas Vector Search

RAGTAG allows you to:

* Create Q/A chunks for use with your chatbots
* Vectorize chunks with open source embedding models (Instructor-large)
* Search existing chunks, edit chunk, update embedding and update chatbot in real time
* Test your chunks for recall by using real questions

## Installation

```pip install -r requirements.txt```

## Downloading the Mistral 7b model (with dolphin fine tune)

```wget https://huggingface.co/TheBloke/dolphin-2.1-mistral-7B-GGUF/resolve/main/dolphin-2.1-mistral-7b.Q5_K_S.gguf```

## Running App

Copy sample.env to .env and modify with connection string to your Atlas instance

```flask run```

## Atlas Search Index

Create and Atlas Search index, in the Atlas UI under the Search tab for the "chunks" collection
under the "ragtag" database.

```
{
  "analyzer": "lucene.english",
  "searchAnalyzer": "lucene.english",
  "mappings": {
    "dynamic": false,
    "fields": {
      "chunk_answer": {
        "type": "string"
      },
      "chunk_embedding": {
        "dimensions": 768,
        "similarity": "cosine",
        "type": "knnVector"
      },
      "chunk_enabled": {
        "type": "boolean"
      },
      "chunk_question": {
        "type": "string"
      }
    }
  }
}
```