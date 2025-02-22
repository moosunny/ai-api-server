from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import app_model
import chat_model

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

model = chat_model.ChatModel()

# @app.get("/chat")
# def chat():
#     response = model.translate()
#     return {response.content}

@app.get("/chat")
def chat():
    try:
        response = model.translate()
        return JSONResponse(content={"bot_response": response})  # JSON 형식으로 반환
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

