from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# model.py를 가져온다.
import model
model = model.AndModel()


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

# /item/{item_id} 경로
# item_id 경로 매개변수 (파라미터)
@app.get("/items/{item_id}") # 엔드 포인트
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/predict/left/{left}/right/{right}") 
def predict(left: int, right: int):
    result = model.predict([left, right])
    return {"result": result}

# 모델의 학습을 요청
@app.post("/train")
def train():
    model.train()
    return {"result": "OK"}


# GPT-2 모델 로드
generator = pipeline("text-generation", model="gpt2")
# JSON Body로 받을 데이터 모델 정의

# pydantic을 활용한 타입 에러 방지
class TextRequest(BaseModel):
    prompt: str
    max_length: int = 100
# POST의 경우 생성형 AI를 활용하기 위해 프롬프트를 JSON 바디로 입력해줘야 함
@app.post("/generate")
async def generate_text(request: TextRequest):
    result = generator(request.prompt, max_length=request.max_length)
    return {"response": result[0]["generated_text"]}