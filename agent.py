from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)
search_results = search.invoke("what is the weather in SF")
print(search_results)
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4", model_provider="openai")

from langchain_core.messages import HumanMessage

from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(model, tools)

