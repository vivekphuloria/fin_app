from langchain.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from pydantic import BaseModel, Field
from typing import Literal 


from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import yfinance as yf



from .consts import l_llm, create_prompt, run_obj_to_dict, run_get_output


## Pydantic Models
class primaryClf(BaseModel):
    """ 
    Classify the input into one or more of the categories
    """ 
    company          : str = Field(description = "Company name in the query")
    ticker           : str = Field(description = "Ticker name of the Query")
    information_type : Literal["General", 'Financial','Market'] = Field(description = 'Type of Information Requested.')
    required_data    : Literal["longName", "longBusinessSummary", "sector", "industry", "website", "fax", "phone", "country", "zip", "city", "address2", "address1",
                               'income_stmt', 'balance_sheet', 'cash_flow', 'quarterly_income_stmt', 'quarterly_balance_sheet', 'quarterly_cash_flow',
                               "history"
                               ] \
        = Field(description = 'List of data Required for fulfilling the query. Allowed values given in the prompt')
    start_date       : str = Field(description = 'Start date of the time requested in user query. YYYY-MM-DD format')
    end_date         : str = Field(description = 'End date of the time requested in user query. YYYY-MM-DD format')


def get_primary_chain(LLM):
    today_str = datetime.today().strftime('%Y-%m-%d')
    month_start = datetime.today().replace(day=1).strftime('%Y-%m-%d')
    month_end = (datetime.today() + relativedelta(day=31)).strftime('%Y-%m-%d')


    primary_prompt = """
You are a equity research analyst. The user query will be about some information regarding a company, and you have to help identify keywords from the user query. 
You have to identify the following items
- 1) "company" : Identify the company the user is talking about. Eg "Tata Consultancy Services", "HDFC Bank", "Aditya Birla Fashion" etc
- 2) "ticker" : The NSE ticker of the identified company. Use Tavily Search to confirm. eg. "TCS.NS", "HDFCBANK.NS", "ABFRL.NS"   
- 3) "information_type" : What type of information the user has requested. The value of this key can only be one of the following "General", "Financial", or "Market". The meaning of these terms is as follows 
        "General" information refers to non-numeric information like its adress, contact information, website, industry, or company description
        "Financial" implies information regarding its balance sheet, cashflows, or PnL statement, either annual or quarterly
        "Market" refers to information regarding its share price
- 4) "required_data": Choose the set of data you need access to. This would be a list of relevant data point , conditional on "information_type" extracted above
    - If "information_type"="General", choose all relevant fields from  "longName", "longBusinessSummary", "sector", "industry", "website", "fax", "phone", "country", "zip", "city", "address2", "address1"
    - If "information_type"="Financial", if the query is requesting quarterly data, choose from ['quarterly_income_stmt', 'quarterly_balance_sheet', 'quarterly_cash_flow'] Else, for annual data choose from ['income_stmt', 'balance_sheet', 'cash_flow']
    - If "information_type"="Market", choose "history" for the stock history

- 5) "start_date" : *Start* Date if the user talking about particular time-duration. Respond keeping in mind that the date today is {today_str}.
This should be expressed in YYYY-MM-DD format, for example for a query talking about this month, start date should be {month_start}
In case of no time mentioned it should be "1900-01-01"

- 6) "start_date" : *End* Date if the user talking about particular time-duration. Respond keeping in mind that the date today is {today_str}.
This should be expressed in YYYY-MM-DD format, for example for a query talking about this month, start date should be {month_end}
In case of no time mentioned it should be "1900-01-31"

Your response has to be a json with these 5 keys - "company", "ticker" "information_type", "required_data", "start_date", "end_date"
Do not have any other information in the response
""".strip().format(today_str=today_str , month_start=month_start, month_end=month_end)

    tool_search = TavilySearchResults(max_results=2)
    l_tools = [tool_search]

    llm_tools = LLM.bind_tools(l_tools)
    llm_st = llm_tools.with_structured_output(primaryClf)
    chain  = create_prompt(primary_prompt) | llm_st | run_obj_to_dict
    return chain



def get_data_response_chain(LLM, d_data, system_prompt):

    # "You have to give response to the users query, given the following information"
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("ai", json.dumps(d_data)),
        ("human", "Query : {user_query}")
    ]
    )
    chain  = prompt | LLM | StrOutputParser()
    return chain


def get_df_agent(LLM, l_df, system_prompt):
    agent = create_pandas_dataframe_agent(
        LLM,
        l_df,
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Query : {user_query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
    )

    chain = prompt | agent | run_get_output
    return chain 

