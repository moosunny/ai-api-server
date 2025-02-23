from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()

search = TavilySearchResults(max_results=2)
search_results = search.invoke("what is the weather in SF")
# print(search_results)
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]
# print(tools)



model = init_chat_model("gpt-4", model_provider="openai")



agent_executor = create_react_agent(model, tools)

# response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})

response = agent_executor.invoke(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
)
response["messages"][-1].pretty_print()

# for step in agent_executor.stream(
#     {"messages": [HumanMessage(content="whats the weather in sf?")]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()

# for step in agent_executor.stream(
#     {"messages": [HumanMessage(content="whats the weather in Seoul?")]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()

