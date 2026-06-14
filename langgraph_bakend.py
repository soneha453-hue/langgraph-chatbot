from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage ,HumanMessage,SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from typing import TypedDict,Annotated

load_dotenv()

llm = ChatGroq(model='openai/gpt-oss-120b')

system_prompt=f"""You are an intelligent AI assistant.

Your behavior:
- Always answer calmly and clearly
- If you don't know something, honestly say "I am sorry, I don't know about this"
- Every answer should be honest and accurate
- Never guess anything — if you don't know, say it openly
- If user writes in Hindi, reply in Hindi
- If user writes in English, reply in English
- If user writes in Hinglish, reply in Hinglish
"""

class chatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


def chat_node(state:chatState):
    messages=state['messages']
    final_message=[SystemMessage(content=system_prompt)]+messages
    response=llm.invoke(final_message)

    return {'messages':[response]}


graph=StateGraph(chatState)
graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

checkpointer=InMemorySaver()

chatbot=graph.compile(checkpointer=checkpointer)

thread_id='1'

# while True:
#     user_msg=input('You : ')

#     if user_msg.strip().lower() in ['exit','bye','quit']:
#         break

#     config={'configurable':{'thread_id':thread_id}}
#     response=chatbot.invoke({'messages':[HumanMessage(content=user_msg)]},config=config)

#     print('Ai : ' ,response['messages'][-1].content)
# # 


