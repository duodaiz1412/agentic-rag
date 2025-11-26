from dotenv import load_dotenv

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from graph.chains.llm_config import create_llm, rate_limit_delay

load_dotenv()

llm = create_llm(model="gemini-2.5-flash", temperature=0)
prompt = hub.pull("rlm/rag-prompt")

base_chain = prompt | llm | StrOutputParser()


def _rate_limited_invoke(input_dict: dict):
    """Wrapper to add rate limiting to generation chain"""
    rate_limit_delay()
    return base_chain.invoke(input_dict)


generation_chain = RunnableLambda(_rate_limited_invoke)
