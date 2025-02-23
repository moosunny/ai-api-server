from fastapi import FastAPI, Query
# from fastapi.responses import StreamingResponse
# from fastapi.staticfiles import StaticFiles

import app_model
import chat_model
import agent_model

app = FastAPI()

model = app_model.AppModel()

@app.get("/say")
def say_app(text: str = Query()): # https:://....../say?text=hi
    response = model.get_response(text)
    return {"contents" :response.content}




@app.get("/translate")
def translate(text: str = Query(), language: str = Query()):
    response =  model.get_prompt_response(language, text)
    return {"contents": response.content}

ch_model = chat_model.ChatModel()


@app.get("/chat")
def chat(text: str = Query(), user: str = Query()):
    response = ch_model.get_response(user, 'English', text)
    return {"content" :response.content}

a_model = agent_model.AgentModel()

@app.get("/search")
def search(user: str = Query(), text: str = Query()):
    response = a_model.get_response(user, text)
    return {"content": response.content}

