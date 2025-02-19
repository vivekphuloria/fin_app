from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from typing import TypedDict, List , Annotated, Dict, Any
from langgraph.graph.message import add_messages



l_llm = {v: ChatOpenAI(model=v) for v in ["gpt-4o","gpt-4o-mini"]}
create_prompt = lambda system_prompt : ChatPromptTemplate.from_messages([('system',system_prompt),('human','{input}')])
run_obj_to_dict = RunnableLambda(lambda x: x.dict())
run_get_output = RunnableLambda(lambda x: x['output'])

# Defining state of Graph

class GraphState(TypedDict):
    """
    Represents the state of the  graph.

    Attributes:
        query: User Query
        company: Name of company in user query
        ticker: Ticker of comapny in user query
        information_type: Type of user query
        required_data: Information required from yfinance API
        time_duration: time_duration extracted from the user query
        generation: LLM generated Response
        messages: List of All Messages

    """

    query: str
    company: str
    ticker:str
    information_type:str
    required_data: List[str]
    time_duration: str
    generation: str
    messages:  Annotated[list, add_messages]
