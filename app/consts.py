from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from typing import TypedDict, List , Annotated, Dict, Any
from langgraph.graph.message import add_messages
from langchain_openai.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv

import os
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
load_dotenv()


# █▀▀ █▀█ █▄░█ █▀ ▀█▀ ▄▀█ █▄░█ ▀█▀ █▀
# █▄▄ █▄█ █░▀█ ▄█ ░█░ █▀█ █░▀█ ░█░ ▄█

l_llm = {v: ChatOpenAI(model=v) for v in ["gpt-4o","gpt-4o-mini","o1-mini"]}
RAG_DOCS_FOL = "C:/Users/vivek.phuloria/Documents/side_projects/fin5/rag_docs"
PINECONE_INDEX_NAME = 'fina-app-feb25'
EMBEDDING_MODEL = OpenAIEmbeddings()
RETRIVER_FETCH_K = 30

tags_struct_info = ["longName", "longBusinessSummary", "sector", "industry", "website", "fax", "phone", "country", "zip", "city", "address2", "address1"]
tags_struct_fin_annual = ['income_stmt', 'balance_sheet', 'cash_flow',] 
tags_struct_fin_quarterly = ['quarterly_income_stmt', 'quarterly_balance_sheet', 'quarterly_cash_flow']
tags_struct_market = ["history"]

d_tags_rag = {
    "Operational Metrics": "Any excerpt that talks about the companies operational metrics for the business, eg. Average Revenue for User, Churn Rate, Net Promoter Score, Resoliton Time etc. Basically anything that indicates how well the company is serving its customers.", 
    "Financial Metrics": "Any excerpt talking about the companies financia; metrics for the business, eg.any Revenue, Costs, Profits, Debts, Valuation, Share Price etc.", 
    "Key Personell": "Any excerpt talking about any executive, management or key personell like a CxO,MD, board of directors etc. - any announcement regarding a change in the personell, modification in their role. or their achievement. ", 
    "Product Strategy": "If the excerpt talks about it's companies portfolio of products or offerings. If any particular product is doing very well or poorly, or if they have changed the product portfolio - adding or removing some items, or changing their focus areas.", 
    "Expansion": "If the excerpt is talking about expansion plans of the organization. This could be by capital expenditure, vertical growth by entering into some other part of the value chain, horizontal growth by targetting a different product line or customer segment, or entering a new business line. This could be an organic growth with internal team working on the new initiative, or inorganic growth via by Mergers or Acquisitions.", 
    "Competition": "If the excerpt talks about it's competitor or the compeitive landscape for its industry.", 
    "Sectoral Phenomenon": "If the excerpt talks about any wholesale changes in consumer preferences; price changes in key ingredients like commodities, energy, labour ; technological advancements ", 
    "Regulatory": "If the excerpt talks about the regulatory landscape, or if the government or any other governing body's policy has impacted the company's business in any way.", 
    "Macro Factors": "If the excerpt talks about any macro phenomena that impacts the business - like global politics, trade treaties, war etc. Also if the excerpt talks about country's economics and populations well being ", 
    "Organizational Details": "If the excerpt talks about the organization structure, or about any parent or subsidiary organization.", 
    "MnA": "If the excerpt talks about any mergers or acquisisions that the company has undertaken.", 
    "Alliances": "If the company has entered into a strategic alliance or partnership with any other organization", 
}




# █▀▀ █░█ █▄░█ █▀▀ ▀█▀ █ █▀█ █▄░█ █▀
# █▀░ █▄█ █░▀█ █▄▄ ░█░ █ █▄█ █░▀█ ▄█

create_prompt = lambda system_prompt,profile='system' : ChatPromptTemplate.from_messages([(profile,system_prompt),('human','{input}')])
run_obj_to_dict = RunnableLambda(lambda x: x.dict())
run_get_output = RunnableLambda(lambda x: x['output'])

## FUNCTIONS FOR Getting METADATA INFO
def get_file_name_metadata(fil_name):
    
    name = fil_name.split('.')[0] # Removing extension
    l_split = name.split('_')
    ticker = l_split[0]+'.NS'
    doc_type = l_split[1]
    dur = l_split[2]


    months = ["Jan", "Feb", "Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    f_mon_str = lambda x: str(months.index(x)+1).rjust(2,"0")

    if len(dur)==5 and dur[:2].isnumeric() and dur[-3:] in months:
        start_date = "20"+dur[:2]+"-"+f_mon_str(dur[-3:])+'-01'
        end_end = (datetime.strptime(start_date, '%Y-%m-%d') + relativedelta(day=31) ).strftime('%Y-%m-%d')
    elif len(dur)==2 and dur.isnumeric():
        start_date = "20"+dur[:2]+"-01-01"
        end_end = "20"+dur[:2]+"-12-31"
    ret = {
        "ticker": ticker,
        "doc_type": doc_type,
        "start_date": start_date,
        "end_date"  : end_end
    }
    return ret


def get_l_dates(rag_docs_fol):
    """
    Used for list of start and end dates. 
    Will be used for filtering dates for fetching RAG documents
    Not a very elgant solution, but required since pinecone doesn't yet support ≥ or ≤ operators on string/date
    """
    l_st_dt, l_end_dt = [],[]
    for fil in os.listdir(rag_docs_fol):
        met = get_file_name_metadata(fil)
        l_st_dt.append(met['start_date'])
        l_end_dt.append(met['end_date'])
    return l_st_dt, l_end_dt


# Defining state of Graph
class GraphState(TypedDict):
    """
    Represents the state of the  graph.

    Attributes:
        query: User Query
        company: Name of company in user query
        ticker: Ticker of comapny in user query
        tags: tags required for filtering excerpts from company filings
        required_data: Information required from yfinance API
        
        start_date: str
        end_date: str

        rag_documents: List[Document]
        list_info: dict
        list_df: List[pd.DataFrame]

        
        generation: LLM generated Response
        messages: List of All Messages

    """
    query: str
    company: str
    ticker:str
    RAG_tags: List[str]
    required_data: List[str]
    start_date: str
    end_date: str
    rag_documents: List[Document]
    list_info: dict
    list_df: List[pd.DataFrame]
    df_analysis:str
    generation: str
    messages:  Annotated[list, add_messages]




## CONSTATNTS RELYING ON ABOVE FUNCTIONS
l_all_st_dates, l_all_end_dates = get_l_dates(RAG_DOCS_FOL)
