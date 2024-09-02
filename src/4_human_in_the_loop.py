import os
from dotenv import load_dotenv

from utils import save_mermaid_to_html
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory,
    # This is new!
    interrupt_before=["tools"],
    # Note: can also interrupt __after__ actions, if desired.
    # interrupt_after=["tools"]
)

# Mermaid記法のGraph図を生成
mermaid_code = graph.get_graph().draw_mermaid()

# Mermaid記法のGraph図をHTMLファイルに出力
save_mermaid_to_html(mermaid_code, "out/human_in_the_loop.html")


# user_input = "I'm learning LangGraph. Could you do some research on it for me?"
user_input = "LangGraphを学んでいます。それについて少し調べ日本語でまとめてもらえますか?"
config = {"configurable": {"thread_id": "1"}}
# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

snapshot = graph.get_state(config)
result = snapshot.next
print(result)

existing_message = snapshot.values["messages"][-1]
result = existing_message.tool_calls
print(result)

# `None` will append nothing new to the current state, letting it resume as if it had never been interrupted
print("")
print("-------------------------- resuming --------------------------")
print("")
events = graph.stream(None, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
