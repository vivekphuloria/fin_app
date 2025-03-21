from consts import GraphState

def primary_node_router(state: GraphState)-> str:
    if state["information_type"] == "General":
        return "YFIN_GENERAL_NODE"
    elif state["information_type"] == "Financial":
        return "YFIN_FIN_NODE"
    elif state["information_type"] == "Market":
        return "YFIN_MARKET_NODE"    
    else:
        return "END"