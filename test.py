import asyncio
from llama_index.llms.openllm import OpenLLM, OpenLLMAPI

remote_llm = OpenLLMAPI(address="http://localhost:3000")

remote_llm.complete('The meaning of life is')


async def main(prompt, **kwargs):
  async for it in remote_llm.astream_chat(prompt, **kwargs):
    print(it)


asyncio.run(main('The time at San Francisco is'))