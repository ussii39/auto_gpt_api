from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import openai
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

app = FastAPI()

# 環境変数の設定
os.environ["SERPAPI_API_KEY"] = ""
openai.api_key = ""

# LLMの初期化
llm = OpenAI(temperature=0.9, openai_api_key=openai.api_key)

# ツールのロード
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# エージェントの初期化
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


class RequestBody(BaseModel):
    query: str


@app.post("/run_agent")
def run_agent(request_body: RequestBody):
    try:
        response = agent.run(request_body.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
