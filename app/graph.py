from dotenv import load_dotenv

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from .consts import GraphState
from .nodes import get_human_node, get_primary_node, get_yf_info_node, get_retrive_node, get_df_analysis_node, get_generation_node
from .edge import primary_node_router



load_dotenv()

def get_graph(): 
    graph =  StateGraph(GraphState)
    # graph.add_node("START", START)
    # graph.add_node("END", END)

    graph.add_node("HUMAN", get_human_node)
    graph.add_node("PRIMARY_CLF", get_primary_node)
    graph.add_node("YFIN_INFO", get_yf_info_node)
    graph.add_node("RAG_RETRIVER", get_retrive_node)
    graph.add_node("DF_ANALYSIS", get_df_analysis_node)
    graph.add_node("GENERATION", get_generation_node)
    
    graph.add_edge(START, "HUMAN")
    graph.add_edge("HUMAN", "PRIMARY_CLF")

    graph.add_edge("PRIMARY_CLF", "RAG_RETRIVER")  ## Paralel routing to retriver and getting yf info
    graph.add_edge("PRIMARY_CLF", "YFIN_INFO")
    graph.add_edge("YFIN_INFO", "DF_ANALYSIS")

    graph.add_edge(["RAG_RETRIVER","DF_ANALYSIS"], "GENERATION")
    graph.add_edge("GENERATION", END)


    # graph.add_conditional_edges("PRIMARY_NODE", primary_node_router) ## Classification node can route to unrelated or retriver node

    # graph.add_edge("YFIN_GENERAL_NODE", END)
    # graph.add_edge("YFIN_FIN_NODE", END)
    # graph.add_edge("YFIN_MARKET_NODE", END)

    
    # memory = SqliteSaver.from_conn_string('fin5_app.sqlite')
    # app = graph.compile(checkpointer=memory)
    
    app = graph.compile()
    
    # Draw graph onto file; Optional  
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")

    return app
