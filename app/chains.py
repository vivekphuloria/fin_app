from langchain.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers.pydantic import PydanticOutputParser

from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from pydantic import BaseModel, Field
from typing import Literal, List


from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import yfinance as yf



from consts import create_prompt, run_obj_to_dict, run_get_output
from consts import tags_struct_info, tags_struct_fin_annual, tags_struct_fin_quarterly, tags_struct_market,d_tags_rag

l_all_struct_tags = tags_struct_info + tags_struct_fin_annual + tags_struct_fin_quarterly + tags_struct_market
l_rag_tags = list(d_tags_rag.keys())

## Pydantic Models
## Pydantic Models
class primaryClf(BaseModel):
    """ 
    Classify the input into one or more of the categories
    """ 
    company          : str = Field(description = "Company name in the query")
    ticker           : str = Field(description = "Ticker name of the Query")
    required_data    : List[Literal[*l_all_struct_tags]] = Field(description = 'List of data Required for fulfilling the query. Allowed values given in the prompt') # type: ignore
    RAG_tags             : List[Literal[*l_rag_tags]] = Field("List of Relevant tags to be searched from the company filings ") # type: ignore
    start_date       : str = Field(description = 'Start date of the time requested in user query. YYYY-MM-DD format')
    end_date         : str = Field(description = 'End date of the time requested in user query. YYYY-MM-DD format')


def get_primary_chain(LLM):
    today_str = datetime.today().strftime('%Y-%m-%d')
    month_start = datetime.today().replace(day=1).strftime('%Y-%m-%d')
    month_end = (datetime.today() + relativedelta(day=31)).strftime('%Y-%m-%d')


    primary_prompt = """
You are helping an equity research analyst. A user will ask a query about some information regarding a company, and you have to help break down the problem by parsing the following items from the query. 
You have to identify the following items
- 1) "company" : Identify the company the user is talking about. Eg "Tata Consultancy Services", "HDFC Bank", "Aditya Birla Fashion" etc
- 2) "ticker" : The NSE ticker of the identified company. Use Tavily Search to confirm. eg. "TCS.NS", "HDFCBANK.NS", "ABFRL.NS"   

- 3) "required_data": The analyst has access to a data source that can provide structured data. 
Choose all the fields that may be relevant in responding to the user query. Your response has to be one or more fields mentioned in "quotes" below
    - If the query would need generic information like its adress, contact information, website, industry, or company description, choose all relevant fields from  {l_info_tags}
    - If the query would need numbers from financial information like  balance sheet, cashflows, or PnL statement,  choose among {l_fin_annual_tags} for annual data, and {l_fin_quarterly_tags} for quarterly data
    - If the query needs the company's market data or share price data, select "history" for the stock history
    There may be cases where the query requests data for ratios like P/E, EV/EBITDA, margins, ROE, ROCE, D/E etc. In these cases, think about all the terms required to calculate it,  which financial statement would contain it, and if stock history data is required, and include all fields in your response.
    The final value of this item, should be a combined list of all fields chosen above
    In case you think none of this information is required for the query, return an empty list - []  


- 4) "RAG_tags" : The analyst has access to tagged passages from the company filings like - Annual Reports, Transcripts, Presentation decks etc.
    Choose all tags whose information would be relevant for answering the user's query. Include as many tags as required.
    List of possible tags: {l_rag_tags}.
    In case you think that the query response needs none of this information from the filings, return an empty list []  


- 5) "start_date" : *Start* Date if the user talking about particular time-duration. Respond keeping in mind that the date today is {today_str}.
This should be expressed in YYYY-MM-DD format, for example for a query talking about this month, start date should be {month_start}
In case of no time mentioned it should be "1900-01-01"

- 6) "start_date" : *End* Date if the user talking about particular time-duration. Respond keeping in mind that the date today is {today_str}.
This should be expressed in YYYY-MM-DD format, for example for a query talking about this month, start date should be {month_end}
In case of no time mentioned it should be "1900-01-31"

Your response has to be a JSON with these keys - "company", "ticker", "required_data", "RAG_tags", "start_date", "end_date"
Do not have any other information in the response
""".strip().format(today_str=today_str , month_start=month_start, month_end=month_end,
                   l_rag_tags = list(d_tags_rag.keys()),
                   l_fin_quarterly_tags = tags_struct_fin_quarterly, 
                   l_fin_annual_tags = tags_struct_fin_annual,
                   l_info_tags = tags_struct_info,
                   )

    
    # llm_st = LLM.with_structured_output(primaryClf)
    # chain  = create_prompt(primary_prompt) | llm_st | run_obj_to_dict

    model_name = LLM.model_name
    system_profile = 'human' if model_name == 'o1-mini' else 'system' 

    parser = PydanticOutputParser(pydantic_object = primaryClf)
    chain  = create_prompt(primary_prompt,profile=system_profile) | LLM | parser | run_obj_to_dict
    return chain


def get_df_analysis_chain(LLM, l_df):

    df_analysis_system_prompt = """ 
    You are a data analysis bot helping a equity research analysts.
    You have been given a user query, and some data about the company's financial performance, and it's share market performane.
    Your primary objective is to provide the equity research analysts with  as many insights about the user-query and any relevant related analysis as possible.
    Do not invent any other information, not provided in the context.
    
    The query may request some ratios, or your analysis may be improved by using some commonly used financial ratios or metrics like Price to Earnings, some margings,  Debt to Equity, ROE, ROCE, EV/EBITDA -  In these cases, identify which fields would be required from which dataframe, join these dataframes if required, and perform the neccessary computations.

    Your response should be a markdown formatted string.
    Provide percentages where applicable. 
    For the stock price or relevant metrics, identify any major positive or negative %age changes, and thir time persiods, and add them to your analysis if relevant to user query. 
    Use you judgement on which metrics and their change and on which period may be most relevant to user query.     

    For currency numbers, use format ₹_ M, B, K for million, billion or thousand in your response.
    Wherever you think a table may be able to showcase your analysis better, please add markdown formatted tables in your response.
    Be structured in your analysis, and help the user understand the business impact of the numbers.

    MOST IMPORTANT INSTRUCTION: Only show analysis that can be directly computed from the data presented to you. Do not, under any circumstance, use any other information outside the data presented to you.   

    """
    agent = create_pandas_dataframe_agent(
        LLM,
        l_df,
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )

    model_name = LLM.model_name
    system_profile = 'human' if  'o1-mini' in model_name  else 'system' 
    prompt = ChatPromptTemplate.from_messages(
    [
        (system_profile, df_analysis_system_prompt),
        ("user", '{input}'),
        ("placeholder", "{agent_scratchpad}"),
    ]
    )

    chain  = prompt | agent | run_get_output
    return chain


def get_generation_chain(LLM):
    prompt =ChatPromptTemplate.from_messages([
        ("system","""
         You are an equity research analyst tasked with answering user queries. Your primary goal is to answer the user query.
         To help you, you've been provided several analysis from your assistants. 
         This may include one or more of the following in your context
         - Extra Information : fetched from company's ticker, 
         - Data Frames : financial statements and / or stock history of the company 
         - DF Analysis: An analyst interpretation of the financial or market data
         - RAG Passages: relevant from the company filings (annual reports, transcripts, PPTs etc)
         
         You are required to synthesise all of this information to answer the user's original query.
         Do not invent any data apart from the above data provided to you in the context
         Be structured and methadological in your response.
         Your answer should be markdown friendly 
        """),
        ("ai","{list_info}"),
        ("ai","{list_df}"),
        ("ai","{df_analysis}"),
        ("ai","{rag_data}"),
        ("human","{query}"),

    ])
    chain =  prompt | LLM | StrOutputParser()
    return chain


# def get_data_response_chain(LLM, d_data, system_prompt):

#     # "You have to give response to the users query, given the following information"
#     prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("ai", json.dumps(d_data)),
#         ("human", '{input}' )
#     ]
#     )
#     chain  = prompt | LLM | StrOutputParser()
#     return chain


# def get_df_agent(LLM, l_df, system_prompt):
#     agent = create_pandas_dataframe_agent(
#         LLM,
#         l_df,
#         verbose=False,
#         agent_type=AgentType.OPENAI_FUNCTIONS,
#         allow_dangerous_code=True
#     )
#     prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", '{input}'),
#         ("placeholder", "{agent_scratchpad}"),
#     ]
#     )

#     chain = prompt | agent | run_get_output
#     return chain 

