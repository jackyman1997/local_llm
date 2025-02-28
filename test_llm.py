from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

llm = Ollama(model="llama3", request_timeout=120.0)

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]

resp = llm.stream_chat(messages)

for r in resp:
    print(r.delta, end="")