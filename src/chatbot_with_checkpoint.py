"""
LangGraphを使って、Web検索を伴うチャットボットを作成します。
会話履歴を保存するために、checkpointを使います。
"""
import os

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

#-- create a MemorySaver checkpointer.
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

#-- Graph の定義
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
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
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# compile
graph = graph_builder.compile(checkpointer=memory)

#-- Now you can interact with your bot! First, pick a thread to use as the key for this conversation.
config = {"configurable": {"thread_id": "1"}}

#-- Next, call your chat bot.
user_input = "LangGraphの最近の主要なアップデートについて教えてください。"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()

#-- Let's ask a followup: see if it remembers your name.
user_input = "アップデートの中から最も重要なものを一つ選んでください"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()

# #-- 
# # The only difference is we change the `thread_id` here to "2" instead of "1"
# events = graph.stream(
#     {"messages": [("user", user_input)]},
#     {"configurable": {"thread_id": "2"}},
#     stream_mode="values",
# )
# for event in events:
#     event["messages"][-1].pretty_print()
