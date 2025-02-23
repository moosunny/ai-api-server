from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

class AgentModel:
    def __init__(self):
        load_dotenv()
        model = init_chat_model("gpt-4", model_provider="openai")
        search = TavilySearchResults(max_results=2)
        tools = [search]

        # memory 설정을 해줘야 하는 부분을 까먹음!
        memory = MemorySaver()
        self.agent_executor = create_react_agent(model, tools, checkpointer= memory)

    def get_response(self, thred_id, message): # tread_id 추가 -> 멀티 유저 고려
        config = {"configurable": {"thread_id": thred_id}}
        response = self.agent_executor.invoke({"messages": [HumanMessage(content=message)]}, config)
        return response["messages"][-1]