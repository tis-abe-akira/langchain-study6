"""
LangGraphを使って、簡単なチャットボットを作成します。
"""
import os

from utils import save_mermaid_to_html

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


# model definition
llm = ChatOpenAI(model="gpt-4o")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()


# Mermaid記法のGraph図を生成
mermaid_code = graph.get_graph().draw_mermaid()

# Mermaid記法のGraph図をHTMLファイルに出力
save_mermaid_to_html(mermaid_code, "out/basic_graph.html")

# Now let's run the chatbot!
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
