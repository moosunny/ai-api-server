from dotenv import load_dotenv
from langchain.chat_models import init_chat_model, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages

from typing import Sequence
from typing_extensions import Annotated, TypedDict

# 커스텀 상태 정의: 언어 입력
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

class ChatModel:
    def __init__(self):
        load_dotenv() 
        self.model = init_chat_model("gpt-4o-mini", model_provider="openai")
        # self.model = ChatOpenAI(model_name = "gpt-4o-mini", temperature = 0.7)
        self.trimmer = trim_messages(max_tokens=65,
                                    strategy="last",
                                    token_counter=self.model, # 오류 발생 가능
                                    include_system=True,
                                    allow_partial=False,
                                    start_on="human",)
        
        self.messages = [SystemMessage(content="you're a good assistant"),
                        HumanMessage(content="hi! I'm bob"),
                        AIMessage(content="hi!"),
                        HumanMessage(content="I like vanilla ice cream"),
                        AIMessage(content="nice"),
                        HumanMessage(content="whats 2 + 2"),
                        AIMessage(content="4"),
                        HumanMessage(content="thanks"),
                        AIMessage(content="no problem!"),
                        HumanMessage(content="having fun?"),
                        AIMessage(content="yes!"),]
        
        # 프롬프트 생성: 언어 번역
        self.prompt_template = ChatPromptTemplate.from_messages(
                            [
                                (
                                    "system",
                                    "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
                                ),
                                MessagesPlaceholder(variable_name="messages"),
                            ]
                        )
                                                        
        self.workflow = StateGraph(state_schema=State)

        # 노드 추가
        self.workflow.add_node("call_model", self.call_model)
        self.workflow.set_entry_point("call_model")

        # Add memory
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

        self.config = {"configurable": {"thread_id": "abc678"}}
        self.query = "What math problem did I ask?"
        self.language = "English"

    # Define the function that calls the model. 인수는 우리가 정의한 State
    def call_model(self, state: State):
        prompt = self.prompt_template.invoke(state)
        response = self.model.invoke(prompt)
        return {"messages": state["messages"] + [AIMessage(content=response)]}
    
    def translate(self):
        self.messages = self.trimmer.invoke(self.messages)
        self.input_messages = self.messages + [HumanMessage(self.query)]
        output = self.app.invoke(
            {"messages": self.input_messages, "language": self.language},
            self.config,
        )
        return output["messages"][-1].pretty_print()
    

        
