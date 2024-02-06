from langchain_community.llms.ctransformers import CTransformers
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from dotenv import load_dotenv
import os
import asyncio


def get_mistral_7b():
    load_dotenv()
    model_id = os.environ['MODEL_ID']
    model_type = os.environ['MODEL_TYPE']
    llm_config = {
        'temperature': 0,
        'context_length': 1028
    }
    return CTransformers(
        model=model_id, 
        model_type=model_type,
        config=llm_config,
        callbacks=[
            AsyncIteratorCallbackHandler()
        ]
    )

def get_agent():
    llm = get_mistral_7b()
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
    asyncio.run(
        call_agent(agent, user_input)
    )