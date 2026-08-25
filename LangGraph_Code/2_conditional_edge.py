from typing import Annotated

from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
class State(BaseModel):
    message:Annotated[str,add_messages("user_input")]
    

    
def node_1(state: State) -> State:
    # print("---Node 1---", state)
    return {"user_input": " i am "}

def node_2(state: State) -> State:
    # print("---Node 2---", state)
    return {"user_input": " happy!"}


def node_3(state: State) -> State:
    print("---Node 3---", state)
    return {"user_input":" sad!"}
def decide(state: State) -> str:
    import random
    random_select=random.uniform(10,100)
    if random_select>2.5: 
        return "node_2"
    else:
        return "node_3"

graph=StateGraph(state_schema=State)
graph.add_node("node_1", node_1)
graph.add_node("node_2", node_2)
graph.add_node("node_3", node_3)

graph.add_edge(START, "node_1")
graph.add_conditional_edges("node_1", decide)
graph.add_edge("node_2", END)
graph.add_edge("node_3", END)
compiled_graph=graph.compile()
user_input=input("Enter your input: ")
result=compiled_graph.invoke({"user_input": user_input})
print(result["user_input"])



