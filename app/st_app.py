import streamlit as st
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage


from graph import get_graph
# from app.consts import HUMAN_NODE

# Will input Langgraph Thread ID, set up app and config variables; set-up session state variables and display older messages in the chat
def set_sidebar():
    with st.sidebar:
        st.header('Configurations')
        st.session_state['thread_input'] = st.text_input(label = "Please enter thread id", value = "FIN_APP-"+datetime.now().strftime("%Y-%m-%d-%H"))
        st.session_state['show_state'] = st.toggle("Show State",value=False)

    pass

def initialize_app():
    # Creating config from that thread-id
    config = {"configurable": {"thread_id": st.session_state['thread_input'] }}
    # App variable
    app = get_graph()

    # Get messages from older config 
    d_map = {HumanMessage: 'user', AIMessage: 'assistant'}
    f_d_map =  lambda l_message: [ {'role':d_map[type(obj)],'content':obj.content} for obj in l_message]
    try:
        st.session_state.messages = f_d_map(app.get_state(config).values['messages'] )
    except: 
        st.session_state.messages = []
    # Show those messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    return app, config




def response_generator(app, config, query):    
    app.update_state( config, {'messages':HumanMessage(query),'query':query}, as_node="HUMAN")
    state = app.invoke(None, config)
    response = state['messages'][-1].content
    return response, state
    


def display_chat_interface():
    st.title("Professional QnA")
    with st.expander("How to use"):
        st.write('''
This is a Q&A app where professional summaries of individuals have been added, and you can ask any relevant questions for that.
Currently only information for Asian Paints for the last 3 years has been added and you can ask any information about that .''')
    
    set_sidebar()
    app, config = initialize_app()



    # Accept user input
    if prompt := st.chat_input("What's your query?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response, graph_state = response_generator(app, config, prompt)
            st.markdown(response)
            if st.session_state['show_state']:
                st.json(graph_state,expanded=False)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    display_chat_interface()