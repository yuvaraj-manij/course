"""
Test context-cap summarization by lowering TOKEN_CAP to 1200 and running
6 pre-scripted turns. Summarization should fire around turn 3-4.
"""
import shutil
import assistant

# Lower cap so summarization fires quickly
assistant.TOKEN_CAP = 500

# Seed a clean store
shutil.rmtree("./chroma_db", ignore_errors=True)
from memory_store import add_fact
add_fact("The user has a daughter named Ava who is 8 years old.")
add_fact("The user works at LifeBridge Insurance as a software engineer.")
add_fact("The user is allergic to peanuts.")
add_fact("The user's dog is a beagle named Bruno.")
print("Store seeded.\n")

turns = [
    ("I want to share some things about myself. I have a sister named Priya who lives in Chennai "
     "with her husband Raj and their two kids. I also have a brother named Vikram who moved to "
     "London three years ago to work in finance. My parents are retired and live in Bangalore — "
     "my dad is 68 and my mom is 65."),
    ("For work, I've been at LifeBridge Insurance for about four years now. I started as a junior "
     "developer but got promoted to senior software engineer last year. My team mostly works on "
     "Python microservices. I've been thinking about transitioning into ML engineering but I'm "
     "not sure if I should get a master's degree first or try the self-study route."),
    ("On the hobbies front, I'm really passionate about street photography — I've been doing it "
     "for about five years. I also started learning Spanish six months ago because I want to "
     "travel to South America next year. My dream trip would be a month in Argentina and Chile."),
    "What do you remember about my family so far? Give me the full picture.",
    ("Given what you know about my background, do you think I should pursue a master's in ML "
     "or try to transition through online courses and self-study?"),
    ("One more thing — my daughter Ava is 8 years old and has started showing interest in coding. "
     "She built her first Scratch project last week and was really excited about it. Do you have "
     "any thoughts on how I can encourage her interest in tech?"),
]

conversation = []
for i, msg in enumerate(turns, 1):
    tok_before = assistant.count_messages_tokens(conversation) if conversation else 0
    print(f"=== Turn {i} (context before: {tok_before} tok) ===")
    print(f"You: {msg}")
    reply = assistant.chat_turn(msg, conversation)
    print(f"Assistant: {reply}")
    print()
