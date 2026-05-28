import shutil
shutil.rmtree("./chroma_db", ignore_errors=True)

from memory_store import add_fact, recall, list_facts

# Scenario 1: add a fact, then a contradicting one
id1 = add_fact("The user has a daughter named Ava.")
print(f"Added Ava-as-daughter: {id1}")

id2 = add_fact("Actually, Ava is the user's niece, not the user's daughter.")
print(f"Added Ava-as-niece: {id2}")

# Inspect store state
print("\n--- All facts (including superseded) ---")
for f in list_facts():
    print(f)

# Recall — should return only the NEWER fact (niece)
print("\n--- recall('Who is Ava?') ---")
for hit in recall("Who is Ava?"):
    print(f"  [{hit['distance']:.3f}] (v{hit['version']}) {hit['fact']}")

# Scenario 2: add a related-but-NOT-contradictory fact
id3 = add_fact("The user has another daughter, Mira.")
# This is RELATED to "Ava is the user's niece" (both about family) but NOT contradictory
print(f"\nAdded Mira: {id3}")

# Scenario 3: add a duplicate (paraphrase)
id4 = add_fact("Ava is a niece of the user.")
print(f"\nDuplicate test: original={id2}, dedup result={id4}  (should be same)")

# Final state
print("\n--- All facts after all inserts ---")
for f in list_facts():
    print(f)
