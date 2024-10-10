import pinecone

pinecone.init(api_key = "YOUR API KEY", enviornment = "YOUR ENVIORNMENT")

pinecone.create_index(name = "insert", dimension = 3)

pinecone.list_indexes()

vectors= [[1, 3, 4], [5, 6, 7], [8, 9, 0]]

vect_ids = ['vec1', 'vec2', 'vec3']

idx = pinecone.Index('insert')

idx.upsert([
    ('vec1', [1, 3, 4]),   
    ('vec2', [5, 6, 7]),
    ('vec3', [8, 9, 0]) 
])  
