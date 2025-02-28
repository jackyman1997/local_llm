from langchain_community.llms.openllm import OpenLLM
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from dotenv import load_dotenv
import os
import asyncio
from llama_index.llms.openllm import OpenLLM as llamaOpenLLM

llm = OpenLLM('HuggingFaceH4/zephyr-7b-alpha')

llm.complete('The meaning of life is')


async def main(prompt, **kwargs):
  async for it in llm.astream_chat(prompt, **kwargs):
    print(it)


asyncio.run(main('The time at San Francisco is'))

def get_llama2_7b_openllm():
    # load_dotenv()
    # model_id = os.environ['MODEL_ID']
    # model_type = os.environ['MODEL_TYPE']
    server_url = "http://localhost:3000"  # Replace with remote host if you are running on a remote server
    llm_config = {
        'temperature': 0,
        'context_length': 1028
    }
    return OpenLLM(
        server_url=server_url,
        server_type='http',
        **llm_config
    )

def get_agent():
    llm = get_llama2_7b_openllm()
    tools = []
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', 'You are an AI assistant.'),
            ('user', '{input}')
        ]
    )
    agent = (
        {
            'input': lambda x: x['input']
        }
        | prompt
        | llm
        # | ReActSingleInputOutputParser()
    )
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )


async def call_agent(agent: AgentExecutor, user_input: dict):
    async for i in agent.astream(user_input): 
        await asyncio.sleep(0.1)
        print(i)


if __name__ == '__main__': 
    user_input = {
        'input': 'tell me a joke'
    }
    agent = get_agent()
    # asyncio.run(
    #     call_agent(agent, user_input)
    # )
    # res = agent(user_input)
    # print(res)
    llm = get_llama2_7b_openllm()
    res = llm('tell me a joke')
    print(res)