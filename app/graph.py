from dotenv import load_dotenv

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from .consts import GraphState
from .nodes import get_human_node, get_primary_node, get_yfin_general_node, get_yfin_fin_node, get_yfin_market_node
from .edge import primary_node_router



load_dotenv()

def get_graph(): 
    graph =  StateGraph(GraphState)
    # graph.add_node("START", START)
    # graph.add_node("END", END)

    graph.add_node("HUMAN_NODE", get_human_node)
    graph.add_node("PRIMARY_NODE", get_primary_node)
    graph.add_node("YFIN_GENERAL_NODE", get_yfin_general_node)
    graph.add_node("YFIN_FIN_NODE", get_yfin_fin_node)
    graph.add_node("YFIN_MARKET_NODE", get_yfin_market_node)
    
    graph.add_edge(START, "HUMAN_NODE")
    graph.add_edge("HUMAN_NODE", "PRIMARY_NODE")    # Both will be called parallely


    graph.add_conditional_edges("PRIMARY_NODE", primary_node_router) ## Classification node can route to unrelated or retriver node

    graph.add_edge("YFIN_GENERAL_NODE", END)
    graph.add_edge("YFIN_FIN_NODE", END)
    graph.add_edge("YFIN_MARKET_NODE", END)

    
    # memory = SqliteSaver.from_conn_string('fin5_app.sqlite')
    # app = graph.compile(checkpointer=memory)
    
    app = graph.compile()
    
    # Draw graph onto file; Optional  
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")

    return app
