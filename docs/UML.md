classDiagram

class RebuildFAISS {
    + rebuild_faiss_index()
}

class RAGSystem {
    - index_file : str
    - metadata_file : str
    - index
    - docs
    - llm
    - client
    + load_vectorstore()
    + rebuild_vectorstore()
    + ask(question, k)
}

class FAISSIndex {
    + add(vectors)
    + search(query, k)
}

class MetadataStore {
    + save()
    + load()
}

class MistralEmbedding {
    + create_embeddings()
}

class ChatMistralAI {
    + invoke(prompt)
}

RebuildFAISS --> MistralEmbedding
RebuildFAISS --> FAISSIndex
RebuildFAISS --> MetadataStore

RAGSystem --> FAISSIndex
RAGSystem --> MetadataStore
RAGSystem --> MistralEmbedding
RAGSystem --> ChatMistralAI
