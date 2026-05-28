import ollama                                              
import chromadb 

# 1. Embeddings via Ollama                                                                                          
def embed(text: str) -> list[float]:                       
    return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

 # Sanity check: same text → same vector; vector length is consistent                                                
v1 = embed("I have a daughter named Ava.")                                                                          
v2 = embed("I have a daughter named Ava.")                                                                          
v3 = embed("My dog is a beagle.")                                                                                   
assert v1 == v2, "embeddings should be deterministic for the same input"
print(f"Vector dim: {len(v1)}")                                                                                     
print(f"v1 == v2: {v1 == v2}")                                                                                      
print(f"v1 == v3: {v1 == v3}  (should be False)")

# 2. Chroma — local, no server                             
client = chromadb.PersistentClient(path="./chroma_db")                                                              
collection = client.get_or_create_collection(                                                                       
    name="facts",
    # we'll provide our own embeddings so Chroma doesn't try to pick a backend                                      
)  

# Wipe any prior state for the smoke test                                                                           
try:                                                       
    collection.delete(ids=collection.get()["ids"])
except Exception:
    pass  

 # 3. Add a few discrete facts with their embeddings                                                                 
facts = [                                                  
    "The user has a daughter named Ava.",
    "The user's dog is a beagle named Bruno.",                                                                      
    "The user works at LifeBridge Insurance as a software engineer.",
    "The user lives in Bangalore, India.",                                                                          
    "The user is allergic to peanuts.",                    
] 

collection.add(                                            
    ids=[f"f{i}" for i in range(len(facts))],                                                                       
    documents=facts,                                       
    embeddings=[embed(f) for f in facts],                                                                           
)

 # 4. Semantic retrieval — query by meaning, not exact match                                                         
queries = [
    "Does the user have kids?",                  # should match Ava fact                                            
    "What kind of pet does the user have?",      # should match Bruno fact                                          
    "Where is the user employed?",               # should match LifeBridge
    "Tell me about the user's allergies.",       # should match peanuts                                             
]  

print("\n--- Retrieval check ---")
for q in queries:                                                                                                   
    res = collection.query(query_embeddings=[embed(q)], n_results=1)
    print(f"Q: {q}")                                                                                                
    print(f"  -> {res['documents'][0][0]}  (distance: {res['distances'][0][0]:.3f})")