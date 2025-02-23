from typing import TypedDict, List , Annotated, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

from datetime import datetime as dt 
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

    
    system_prompt = """
    You are a bot helping users get general information about company's contact information, industry, business overview etc. 
    Answer the users query, given the following information. Do not invent any other information, not provided in the context.
    Keep the answer professional, concise and to the point.
    """.strip()

    chain = get_data_response_chain(LLM=l_llm['gpt-4o'], d_data=d_data, system_prompt=system_prompt)
    
    res = chain.invoke(state['query'])
    return {"generation" : res}


def get_yfin_fin_node(state: GraphState)-> Dict[str, Any]:
    stock = yf.Ticker(state['ticker'])
    
    l_df = []
    for field in state["required_data"]:
        l_df.append(getattr(stock,field))

    
    fin_system_prompt = """
    You are a bot helping equity research analysts get insigths about company's financial performance. 
    Your job is to answer the users query, given the following information. Do not invent any other information, not provided in the context.
    Your response should be a markdown formatted string.
    Provide percentages where applicable. Use you judgement on which fields may be most important for analysing stocks.
    For currency numbers, use format ₹_ M, B, K for million, billion or thousand in your response.
    Do not mention that a dataframe was provided to you, instead show it as your analysis.
    Wherever you think a table may be relevant, you can add markdown formatted tables in your response.
    Be structured in your analysis, and help the user understand the business impact of the numbers. 
    """

    chain = get_df_agent(LLM=l_llm['gpt-4o'], l_df=l_df, system_prompt=fin_system_prompt)
    
    res = chain.invoke(state['query'])
    return {"generation" : res}

def get_yfin_market_node(state: GraphState)-> Dict[str, Any]:
    start_date = dt.strptime(state['start_date'],'%Y-%M-%d')
    end_date = dt.strptime(state['end_date'],'%Y-%M-%d')
    total_interval = (start_date - end_date).days
    if total_interval < 95: # Upto a little over 1 quarter
        hist_interval = '1d' 
    if total_interval < 400: # Upto a little over 1 year
        hist_interval = '1wk' 
    else:  # Over an year
        hist_interval = '1mo'
    stock = yf.Ticker(state['ticker'])

    hist = stock.history(start=start_date, end = end_date, interval=hist_interval)
    hist['percentage_change'] = hist.pct_change()['Close']
    mkt_system_prompt = """
    You are a bot helping equity research analysts get insigths about company's stock price variations. 
    Your job is to answer the users query, given the following information. Do not invent any other information, not provided in the context.    
    Identify major positive or negative %age changes, and thir time persiods, and add them to your analysis if relevant to user query. 
    Provide percentages where applicable.
    Use you judgement on which change and period may be most relevant to user query.     
    Be structured in your analysis, and help the user understand the business impact of the numbers. 
    """

    chain = get_df_agent(LLM=l_llm['gpt-4o'], l_df=hist, system_prompt=mkt_system_prompt)
    
    res = chain.invoke(state['query'])    

    return {"generation" : res}
