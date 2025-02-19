from typing import TypedDict, List , Annotated, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

import yfinance as yf
from .consts import GraphState, l_llm
from .chains import get_primary_chain, get_data_response_chain , get_df_agent




def get_human_node(state: GraphState)-> Dict[str, Any]:
    return {'messages':HumanMessage(state['query'])} 

def get_primary_node(state: GraphState)-> Dict[str, Any]:
    chain = get_primary_chain(LLM=l_llm['gpt-4o'])
    res = chain.invoke(state['query'])
    
    return res


def get_yfin_general_node(state: GraphState)-> Dict[str, Any]:
    stock = yf.Ticker(state['ticker'])
    
    d_data = {}
    for field in state["required_data"]:
        if field in stock.info.keys():
            d_data[field] = stock.info[field]

    
    system_prompt = "You have to give response to the users query, given the following information"

    chain = get_data_response_chain(LLM=l_llm['gpt-4o'], d_data=d_data, system_prompt=system_prompt)
    
    res = chain.invoke(state['query'])
    return {"generation" : res}


def get_yfin_fin_node(state: GraphState)-> Dict[str, Any]:
    stock = yf.Ticker(state['ticker'])
    
    l_df = []
    for field in state["required_data"]:
        l_df.append(getattr(stock,field))

    
    system_prompt = "You have to give response to the users query, given the following information"

    chain = get_df_agent(LLM=l_llm['gpt-4o'], l_df=l_df, system_prompt=system_prompt)
    
    res = chain.invoke(state['query'])
    return {"generation" : res}

def get_yfin_market_node(state: GraphState)-> Dict[str, Any]:
    ## ADD PROPER LOGIC
    return {"generation" : "MARKET LOGIC TO BE ADDED"}
