pip install pinecone
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

PINECONE_API_KEY="YOUR API KEY"
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="YOUR API KEY")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
index_name = "quickstart1"

pc.create_index(
    name=index_name,
    dimension=1024, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
data = [
    {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
    {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
    {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
    {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
    {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
    {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
]
#Converting text data into embeddings
embeddings = pc.inference.embed(
    model="multilingual-e5-large", #pre-trained model
    inputs=[d['text'] for d in data],
    parameters={"input_type": "passage", "truncate": "END"}
)
print(embeddings[0])

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# This code snippet is designed to wait for a Pinecone index to be ready and then upsert (update or insert) a list of vectors (embeddings) into that index

# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

#Initialize the Index
index = pc.Index(index_name)

#Prepare Vecotrs for upserting
vectors = []
for d, e in zip(data, embeddings):
    vectors.append({
        "id": d['id'],
        "values": e['values'],
        "metadata": {'text': d['text']}
    })

index.upsert(
    vectors=vectors,
    namespace="ns1"
)

# This process is crucial for setting up a searchable vector database for tasks such as similarity search or semantic retrieval.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

print(index.describe_index_stats())


# create an embedding for a query string using Pinecone's API and the multilingual-e5-large model.

query = "Tell me about the tech company known as Apple."
embedding = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[query],
    parameters={
        "input_type": "query"
    }
)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# This code snippet takes a natural language query about the tech company Apple and generates a high-dimensional vector 
# embedding for that query using the specified multilingual embedding model. This embedding can then be used for tasks such as 
# similarity search in a vector database, allowing the system to find relevant results based on the query's semantic meaning.


results = index.query(
    namespace="ns1",
    vector=embedding[0].values,
    top_k=3,
    include_values=False,
    include_metadata=True
)

print(results)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
