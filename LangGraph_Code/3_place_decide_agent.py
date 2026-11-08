from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import os
from dotenv import load_dotenv

load_dotenv()
whether_api = os.getenv("whether_api_key")
tavily_api = os.getenv("tavily_api_key")

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=os.getenv("api_key"))


class ExtractedDetails(BaseModel):
    city: str = Field(description="The destination city specified by the user.")
    budget_level: str = Field(description="The budget preference (e.g., budget, moderate, luxury).", default="moderate")
    specific_food_dish: Optional[str] = Field(description="Specific food, dish, or cuisine requested by the user (e.g., biryani, nihari, pizza).", default=None)


class PlaceDecideState(MessagesState):
    city: str
    budget_level: str
    specific_food_dish: Optional[str]
    weather_data: str
    tourist_data: List[str]
    budget_data: str
    accommodation_data: str
    cuisine_data: str
    final_itinerary: str


def input_analyzer_node(state: PlaceDecideState):
    """Analyzes the user message to extract destination city, budget preferences, and specific food requests."""
    structured_llm = llm.with_structured_output(ExtractedDetails)
    system_prompt = SystemMessage(content="Extract the travel destination city, budget preference, and any specific food or dish requested by the user (e.g., biryani, nihari, etc.).")
    extracted = structured_llm.invoke([system_prompt] + state["messages"])
    return {
        "city": extracted.city,
        "budget_level": extracted.budget_level,
        "specific_food_dish": extracted.specific_food_dish
    }


def weather_info_node(state: PlaceDecideState):
    """Gets weather details for the target city."""
    city = state.get("city", "Multan")
    if whether_api:
        response = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={whether_api}")
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp'] - 273.15
            description = data['weather'][0]['description']
            return {"weather_data": f"Weather in {city}: {temp:.1f}°C with {description}."}
    
    # Fallback to LLM weather estimate
    res = llm.invoke([HumanMessage(content=f"Provide expected seasonal weather details for {city}.")])
    return {"weather_data": res.content}


def tourist_info_node(state: PlaceDecideState):
    """Gets top tourist attractions for the target city."""
    city = state.get("city", "Multan")
    if tavily_api:
        tavily_search = TavilySearchResults(api_key=tavily_api, max_results=3)
        result = tavily_search.invoke({"query": f"top tourist places to visit in {city}"})
        tourist_results = []
        for idx, item in enumerate(result):
            tourist_results.append(f"{idx+1}. {item['title']}: {item['content']}")
        return {"tourist_data": tourist_results}
    
    res = llm.invoke([HumanMessage(content=f"List top 3 tourist attractions in {city}.")])
    return {"tourist_data": [res.content]}


def budget_info_node(state: PlaceDecideState):
    """Evaluates cost estimates and budget constraints."""
    city = state.get("city", "Multan")
    budget = state.get("budget_level", "moderate")
    res = llm.invoke([HumanMessage(content=f"Provide a cost estimate and breakdown for a {budget} trip to {city}.")])
    return {"budget_data": res.content}


def accommodation_info_node(state: PlaceDecideState):
    """Provides stay and hotel recommendations."""
    city = state.get("city", "Multan")
    budget = state.get("budget_level", "moderate")
    res = llm.invoke([HumanMessage(content=f"Recommend 2-3 hotel or stay options for a {budget} budget in {city}.")])
    return {"accommodation_data": res.content}


def cuisine_info_node(state: PlaceDecideState):
    """Recommends local foods and specific places/restaurants known for requested food or dishes."""
    city = state.get("city", "Multan")
    specific_food = state.get("specific_food_dish")
    
    if specific_food:
        query = f"Best places and famous restaurants to eat {specific_food} in {city}"
        if tavily_api:
            tavily_search = TavilySearchResults(api_key=tavily_api, max_results=3)
            result = tavily_search.invoke({"query": query})
            food_results = []
            for idx, item in enumerate(result):
                food_results.append(f"{idx+1}. {item['title']}: {item['content']}")
            search_context = "\n".join(food_results)
            res = llm.invoke([HumanMessage(content=f"Recommend the best spots to eat {specific_food} in {city} based on search info:\n{search_context}")])
            return {"cuisine_data": res.content}
        else:
            res = llm.invoke([HumanMessage(content=f"Recommend the top-rated restaurants and places in {city} known specifically for serving the best {specific_food}.")])
            return {"cuisine_data": res.content}
            
    res = llm.invoke([HumanMessage(content=f"Recommend famous local foods, dishes, and dining spots in {city}.")])
    return {"cuisine_data": res.content}


def final_itinerary_node(state: PlaceDecideState):
    """Aggregates outputs into a comprehensive final travel itinerary speaking in Siraiki (Roman English)."""
    city = state.get("city", "Multan")
    specific_food = state.get("specific_food_dish", "")
    weather = state.get("weather_data", "")
    tourist = "\n".join(state.get("tourist_data", []))
    budget = state.get("budget_data", "")
    accommodation = state.get("accommodation_data", "")
    cuisine = state.get("cuisine_data", "")

    system_prompt = (
        "You are an expert travel assistant. Synthesize all information (weather, tourist spots, budget, "
        "accommodations, and cuisine/food spot recommendations) into a comprehensive travel recommendation and day itinerary. "
        "If a specific dish or food item was requested, make sure to highlight places offering the best version of that food. "
        "Talk in Siraiki language using Roman English script."
    )

    content = f"""
City: {city}
Requested Food/Dish: {specific_food if specific_food else "N/A"}
Weather Info: {weather}
Tourist Attractions:
{tourist}
Budget Details: {budget}
Accommodations: {accommodation}
Local Cuisine & Special Food Places: {cuisine}
"""
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=content)])
    return {
        "final_itinerary": response.content,
        "messages": [AIMessage(content=response.content)]
    }


def route_user_preference(state: PlaceDecideState) -> List[str]:
    """Dynamically routes from input analyzer to all detail collection nodes in parallel."""
    return [
        "weather_info_node",
        "tourist_info_node",
        "budget_info_node",
        "accommodation_info_node",
        "cuisine_info_node"
    ]


# Graph construction
graph = StateGraph(PlaceDecideState)

graph.add_node("input_analyzer_node", input_analyzer_node)
graph.add_node("weather_info_node", weather_info_node)
graph.add_node("tourist_info_node", tourist_info_node)
graph.add_node("budget_info_node", budget_info_node)
graph.add_node("accommodation_info_node", accommodation_info_node)
graph.add_node("cuisine_info_node", cuisine_info_node)
graph.add_node("final_itinerary_node", final_itinerary_node)

graph.add_edge(START, "input_analyzer_node")
graph.add_conditional_edges("input_analyzer_node", route_user_preference)

# Fan-in edges to final_itinerary_node
graph.add_edge("weather_info_node", "final_itinerary_node")
graph.add_edge("tourist_info_node", "final_itinerary_node")
graph.add_edge("budget_info_node", "final_itinerary_node")
graph.add_edge("accommodation_info_node", "final_itinerary_node")
graph.add_edge("cuisine_info_node", "final_itinerary_node")

graph.add_edge("final_itinerary_node", END)

compile_graph = graph.compile()

if __name__ == "__main__":
    user_query = input("Enter your trip query (e.g., I want to visit Multan on a luxury budget): ")
    result = compile_graph.invoke({"messages": [HumanMessage(content=user_query)]})
    print("\n--- FINAL ITINERARY ---")
    print(result.get("final_itinerary"))