from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

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
        
        # Initialize chat history with session ID and database connection.
        chat_message_history = SQLChatMessageHistory(session_id="sql_history", connection="sqlite:///sqlite.db")                                                    
        
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

# from dotenv import load_dotenv
# from langchain.chat_models import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, trim_messages
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_community.chat_message_histories import SQLChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.output_parsers import StrOutputParser

# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import START, StateGraph
# from langgraph.graph.message import add_messages

# from typing import Sequence
# from typing_extensions import Annotated, TypedDict


# # 커스텀 상태 정의
# class State(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     language: str
#     session_id: str


# class ChatModel:
#     def __init__(self, thread_id="default_session"):
#         load_dotenv() 

#         # OpenAI 모델 초기화
#         self.model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

#         # 메시지 길이 제한을 위한 Trimmer 설정
#         self.trimmer = trim_messages(
#             max_tokens=65,
#             strategy="last",
#             token_counter=self.model,  # 오류 발생 가능, 필요 시 수정
#             include_system=True,
#             allow_partial=False,
#             start_on="human",
#         )

#         # 프롬프트 템플릿 생성
#         self.prompt_template = ChatPromptTemplate.from_messages([
#             ("system", "You are a helpful assistant. Answer all questions to the best of your ability in {language}."),
#             MessagesPlaceholder(variable_name="messages"),
#         ])

#         # 데이터베이스 연결
#         self.db_connection = "sqlite:///sqlite.db"

#         # 세션 ID 설정
#         self.session_id = thread_id

#         # 기존 대화 이력 불러오기
#         self.chat_history = SQLChatMessageHistory(session_id=self.session_id, connection=self.db_connection)
#         self.messages = self.chat_history.messages  # 기존 메시지 로드

#         # LangGraph Workflow 생성
#         self.workflow = StateGraph(State)

#         # 모델 호출 함수 정의
#         def call_model(state: State):
#             # SQL 데이터베이스에서 메시지 기록 가져오기
#             chat_history = SQLChatMessageHistory(session_id=state["session_id"], connection=self.db_connection)
#             messages = chat_history.messages + state["messages"]  # 기존 메시지 + 새로운 메시지

#             # 메시지 정리
#             trimmed_messages = self.trimmer.invoke(messages)

#             # 프롬프트 생성
#             prompt = self.prompt_template.invoke({
#                 "messages": trimmed_messages,
#                 "language": state["language"]
#             })

#             # 모델 호출
#             response = self.model.invoke(prompt)
#             response_message = AIMessage(content=response.content)

#             # 데이터베이스에 메시지 저장
#             chat_history.add_user_message(state["messages"][-1].content)  # 사용자 메시지 저장
#             chat_history.add_ai_message(response_message.content)  # AI 응답 저장

#             return {
#                 "messages": messages + [response_message], 
#                 "language": state["language"], 
#                 "session_id": state["session_id"]
#             }

#         # 노드 추가
#         self.workflow.add_node("call_model", call_model)
#         self.workflow.set_entry_point("call_model")

#         # 메모리 체크포인트 추가
#         memory = MemorySaver()

#         # RunnableWithMessageHistory 적용 (세션 ID 관리)
#         self.app = RunnableWithMessageHistory(
#             self.workflow.compile(checkpointer=memory),
#             lambda session_id: SQLChatMessageHistory(session_id=session_id, connection=self.db_connection),
#             input_messages_key="messages",
#             history_messages_key="messages"
#         )

#     def get_response(self, message: str, language: str):
#         config = {"configurable": {"thread_id": self.session_id}}

#         # 입력 메시지 구성 (과거 대화 포함)
#         input_messages = self.messages + [HumanMessage(content=message)]

#         # LangGraph 실행
#         output = self.app.invoke(
#             {"messages": input_messages, "language": language, "session_id": self.session_id},
#             config
#         )

#         # 최신 메시지 업데이트
#         self.messages = output["messages"]

#         return output["messages"][-1]  # 마지막 AI 응답 반환
