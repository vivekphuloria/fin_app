from typing import TypedDict, List , Annotated, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langchain_pinecone import PineconeVectorStore

from datetime import datetime
import pandas as pd
from io import StringIO  
import yfinance as yf

from consts import PINECONE_INDEX_NAME, EMBEDDING_MODEL, RETRIVER_FETCH_K
from consts import GraphState, l_llm, l_all_end_dates, l_all_st_dates
from consts import tags_struct_info, tags_struct_fin_annual, tags_struct_fin_quarterly, tags_struct_market

from chains import get_primary_chain, get_df_analysis_chain, get_generation_chain
import json

def get_human_node(state: GraphState)-> Dict[str, Any]:
    return {'messages':HumanMessage(state['query'])} 

def get_primary_node(state: GraphState)-> Dict[str, Any]:
    chain = get_primary_chain(LLM=l_llm['o1-mini'])
    res = chain.invoke({"input": state['query']})

    # print('---EXITING PRIMARY CLF---')
    return res

def get_retrive_node(state: GraphState) -> Dict[str, Any] :
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=EMBEDDING_MODEL)

    # For retrival, only metadata_tags can be used. Only selecting those from the classification
    # metadata_tag_filters = list(set(state['classification']).intersection(set(METADATA_TAGS)))
    
    ## Filtered sets will be all start dates before the required end-date, and all end-dates after the required start dates
    l_filtered_st_date  = [i for i in l_all_st_dates if i <= state["end_date"]]
    l_filtered_end_date = [i for i in l_all_end_dates if i >= state["start_date"]]

    full_metadata_filter = { 
        "$and": 
        [
            {"tag": {"$in":state["RAG_tags"]}},
            {"ticker": state['ticker'] },
            {"start_date": {"$in":l_filtered_st_date}},
            {"end_date"  : {"$in":l_filtered_end_date}},
      ] 
     }
    retriver = vectorstore.as_retriever(
        search_kwargs= {
            "filter": full_metadata_filter,
            "k": RETRIVER_FETCH_K})
    docs = retriver.invoke(state['query'])
    # print('---EXITING RETRIVER---')
    return {'rag_documents':docs}





def get_yf_info_node(state):
    req_data = state['required_data']
    
    if len(req_data)==0:
        return {}
    else:
        stock = yf.Ticker(state['ticker'])

    intersection = lambda a,b : list(set(a) & set(b))


    l_fields_req_info = intersection(req_data, tags_struct_info)
    l_fields_req_fin = intersection(req_data, tags_struct_fin_annual+tags_struct_fin_quarterly)
    l_fields_req_market = intersection(req_data, tags_struct_market)

    d_info = {}
    l_df = []


    if len(l_fields_req_info)>0:
        for field in l_fields_req_info:
            if field in stock.info.keys():
                d_info[field] = stock.info[field]
    
    if len(l_fields_req_fin) > 0:
        for field in l_fields_req_fin:
            l_df.append(getattr(stock,field))
    
    if len(l_fields_req_market)>0:
        start_date = datetime.strptime(state['start_date'],'%Y-%M-%d')
        end_date = datetime.strptime(state['end_date'],'%Y-%M-%d')
        total_interval = (start_date - end_date).days
        if total_interval < 95: # Upto a little over 1 quarter
            hist_interval = '1d' 
        if total_interval < 400: # Upto a little over 1 year
            hist_interval = '1wk' 
        else:  # Over an year
            hist_interval = '1mo'

        hist = stock.history(start=start_date, end = end_date, interval=hist_interval)
        hist['percentage_change'] = hist.pct_change()['Close']

        l_df.append(hist)
    
    # Changing DF to csv strings so that they can be serialized for memmory
    l_df_csv = []
    for df in l_df:
        l_df_csv.append(df.to_csv(index = False))

    # print('---EXITING YFIN INFO---')
    return {"list_info": d_info, "list_df": l_df_csv}


def get_df_analysis_node(state: GraphState) -> Dict[str, Any] :
    l_df_obj = [pd.read_csv(StringIO(df_csv)) for df_csv in state['list_df']]

    chain = get_df_analysis_chain(LLM=l_llm['gpt-4o'], l_df=l_df_obj)
    res = chain.invoke({"input": state['query']})

    # print('---EXITING DF ANALSIS---')
    return {"df_analysis": res} 



def get_generation_node(state: GraphState) -> Dict[str, Any] :
    gen_chain = get_generation_chain(LLM=l_llm['gpt-4o'])
    gen = gen_chain.invoke( input = {
        'query' : state["query"],
        'list_info' : json.dumps(state["list_info"]),
        'list_df' : "\n\n------------\n\n".join(pd.read_csv(StringIO(df)).to_markdown() for df in state['list_df']),
        'df_analysis' : state['df_analysis'],
        'rag_data' : "\n--------\n".join(doc.page_content for doc in state['rag_documents'])
        }
    )   
    return {"generation": gen, "messages": AIMessage(gen) } 



def get_nodes() -> Dict:
    """
    return dictionary of 
    """
    d_nodes = {
    "HUMAN" :  get_human_node,
    "PRIMARY_CLF" :  get_primary_node,
    "YFIN_INFO" :  get_yf_info_node,
    "RAG_RETRIVER" :  get_retrive_node,
    "DF_ANALYSIS" :  get_df_analysis_node,
    "GENERATION" :  get_generation_node,
    }
    return d_nodes



# def get_yfin_general_node(state: GraphState)-> Dict[str, Any]:
#     stock = yf.Ticker(state['ticker'])
    
#     d_data = {}
#     for field in state["required_data"]:
#         if field in stock.info.keys():
#             d_data[field] = stock.info[field]

    
#     system_prompt = """
#     You are a bot helping users get general information about company's contact information, industry, business overview etc. 
#     Answer the users query, given the following information. Do not invent any other information, not provided in the context.
#     Keep the answer professional, concise and to the point.
#     """.strip()

#     chain = get_data_response_chain(LLM=l_llm['gpt-4o'], d_data=d_data, system_prompt=system_prompt)
    
#     res = chain.invoke(state['query'])
#     return {"generation" : res}


# def get_yfin_fin_node(state: GraphState)-> Dict[str, Any]:
#     stock = yf.Ticker(state['ticker'])
    
#     l_df = []
#     for field in state["required_data"]:
#         l_df.append(getattr(stock,field))

    
#     # The following was a great for a final-response generation use-case; Need to chage if need it just for insights, and generation is led later  
#     fin_system_prompt = """ 
#     You are a bot helping equity research analysts get insigths about company's financial performance. 
#     Your job is to answer the users query, given the following information. Do not invent any other information, not provided in the context.
#     Your response should be a markdown formatted string.
#     Provide percentages where applicable. Use you judgement on which fields may be most important for analysing stocks.
#     For currency numbers, use format ₹_ M, B, K for million, billion or thousand in your response.
#     Do not mention that a dataframe was provided to you, instead show it as your analysis.
#     Wherever you think a table may be relevant, you can add markdown formatted tables in your response.
#     Be structured in your analysis, and help the user understand the business impact of the numbers. 
#     """


#     chain = get_df_agent(LLM=l_llm['gpt-4o'], l_df=l_df, system_prompt=fin_system_prompt)
    
#     res = chain.invoke(state['query'])
#     return {"generation" : res}

# def get_yfin_market_node(state: GraphState)-> Dict[str, Any]:
#     start_date = datetime.strptime(state['start_date'],'%Y-%M-%d')
#     end_date = datetime.strptime(state['end_date'],'%Y-%M-%d')
#     total_interval = (start_date - end_date).days
#     if total_interval < 95: # Upto a little over 1 quarter
#         hist_interval = '1d' 
#     if total_interval < 400: # Upto a little over 1 year
#         hist_interval = '1wk' 
#     else:  # Over an year
#         hist_interval = '1mo'
#     stock = yf.Ticker(state['ticker'])

#     hist = stock.history(start=start_date, end = end_date, interval=hist_interval)
#     hist['percentage_change'] = hist.pct_change()['Close']
#     mkt_system_prompt = """
#     You are a bot helping equity research analysts get insigths about company's stock price variations. 
#     Your job is to answer the users query, given the following information. Do not invent any other information, not provided in the context.    
#     Identify major positive or negative %age changes, and thir time persiods, and add them to your analysis if relevant to user query. 
#     Provide percentages where applicable.
#     Use you judgement on which change and period may be most relevant to user query.     
#     Be structured in your analysis, and help the user understand the business impact of the numbers. 
#     """

#     chain = get_df_agent(LLM=l_llm['gpt-4o'], l_df=hist, system_prompt=mkt_system_prompt)
    
#     res = chain.invoke(state['query'])    

#     return {"generation" : res}
