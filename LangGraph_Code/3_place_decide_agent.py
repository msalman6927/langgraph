from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import Annotated, MessagesState,add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import os
from dotenv import load_dotenv
load_dotenv()
whether_api=os.getenv("whether_api_key")
tavily_api=os.getenv("tavily_api_key")

llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash",google_api_key=os.getenv("api_key"))

def weather_info(city:str):
    """
    this is function to get weather information of a city"""
    response = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={whether_api}")
    if response.status_code == 200:
        data=response.json()
        temperature=data['main']['temp']
        description=data['weather'][0]['description']
        return f"The temperature in {city} is {temperature}°C with {description}."
    
    
def tourist_info(query:str)->dict:
    """
    this is function to get tourist information of a city
    
    args:
        query (str): The name of the city for which to retrieve tourist information.
    
    returns:
        return information about the tourist attractions, activities, and points of interest in the specified city.
    """
    if tavily_api:
        tavily_search=TavilySearchResults(api_key=tavily_api,max_results=3)
        result=tavily_search.invoke({"query":query})
        tourist_results=[]
        for idx,item in enumerate(result):
            index=idx+1
            title=item["title"]
            description=item["content"]
            tourist_results.append(f"{index}. {title}: {description}")
        return {
                "places_data":tourist_results
            }
    
    else:
        return {
            "places_data":f"tavily_api_key is not provided in .env file"
        }
tools=[weather_info,tourist_info]
llm_with_tools=llm.bind_tools(tools)

def llm_node(state:MessagesState):
    system_prompt="You are a helpful assistant that provides information about weather and tourist attractions in different cities.u ahve to talk in siraiki in roman english language. "

    response=llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    
    return {"messages":response}


tool=ToolNode(tools)


graph=StateGraph(state_schema=MessagesState)
graph.add_node("llm_node", llm_node)
graph.add_node("tools", tool)

graph.add_edge(START, "llm_node")
graph.add_conditional_edges("llm_node", tools_condition)
graph.add_edge("tools", "llm_node")

compile_graph=graph.compile()

user=input("Enter your query:")
result=compile_graph.invoke({"messages": [HumanMessage(content=user)]})
for res in result["messages"]:
    res.pretty_print()