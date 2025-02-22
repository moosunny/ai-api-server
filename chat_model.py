from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
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
        model = init_chat_model("gpt-4o-mini", model_provider="openai")
        # self.model = ChatOpenAI(model_name = "gpt-4o-mini", temperature = 0.7)
        trimmer = trim_messages(max_tokens=65,
                                    strategy="last",
                                    token_counter=model, # 오류 발생 가능
                                    include_system=True,
                                    allow_partial=False,
                                    start_on="human",)
        
        # 프롬프트 생성: 언어 번역
        prompt_template = ChatPromptTemplate.from_messages(
                            [
                                (
                                    "system",
                                    "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
                                ),
                                MessagesPlaceholder(variable_name="messages"),
                            ]
                        )
        

        
            # Define the function that calls the model. 인수는 우리가 정의한 State
        def call_model(state: State):
            trimmed_messages = trimmer.invoke(state["messages"])
            prompt = prompt_template.invoke({
                "messages":trimmed_messages, "language": state["language"]})
            response = model.invoke(prompt)
            return {"messages": response}
        

                                                        
        workflow = StateGraph(state_schema=State)

        # 노드 추가
        workflow.add_node("call_model", call_model)
        workflow.set_entry_point("call_model")

        # Add memory
        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)

        # self.config = {"configurable": {"thread_id": "abc678"}}
        # self.query = "What math problem did I ask?"
        # self.language = "English"
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

    
    def get_response(self, thred_id, language, message):
        config = {"configurable": {"thread_id": thred_id}}
        input_messages = self.messages + [HumanMessage(message)]
        output = self.app.invoke(
            {"messages": input_messages, "language": language},
            config,
        )
        return output["messages"][-1] # pretty_print()는 NoneTyoe 오류 발생