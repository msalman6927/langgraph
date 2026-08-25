
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage,AIMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
import os

YOUR_API_KEY=os.getenv("api_key")
class MyState(BaseModel):
    input:str
    output:str
    
llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash",api_key=YOUR_API_KEY)
def llm_node(state:MyState):
    message=HumanMessage(content=state.input)
    response=llm.invoke([message])
    return {"output":response.content}

graph=StateGraph(state_schema=MyState)
graph.add_node("llm_node",llm_node)
graph.add_edge(START,"llm_node")
graph.add_edge("llm_node",END)

compile_graph=graph.compile()
user_input=input("Enter your input: ")
result=compile_graph.invoke({"input": user_input,"output": ""})
print(result["output"][0]["text"])
