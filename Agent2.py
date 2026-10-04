# 1. IMPORTS

from dotenv import load_dotenv
import os
import requests
from rich import print

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from langchain_groq import ChatGroq
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call


# 2. LOAD ENVIRONMENT VARIABLES

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

print("OpenWeather API key exists:", OPENWEATHER_API_KEY is not None)
print(
    "OpenWeather API key length:",
    len(OPENWEATHER_API_KEY) if OPENWEATHER_API_KEY else 0
)



# 3. WEATHER TOOL

@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city in Pakistan."""

    if not OPENWEATHER_API_KEY:
        return "OpenWeather API key is missing."

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": f"{city},PK",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params)

    safe_url = response.url.replace(
        OPENWEATHER_API_KEY,
        "HIDDEN"
    )

    print("Request URL:", safe_url)

    if response.status_code != 200:
        return f"OpenWeather error {response.status_code}: {response.text}"

    data = response.json()

    return str({
        "city": data["name"],
        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "description": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"]
    })


# 4. TAVILY SEARCH CLIENT


tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)


# 5. CITY NEWS TOOL

@tool
def search_city_news(city: str) -> str:
    """Search for the latest news about a specific city in Pakistan."""

    response = tavily_client.search(
        query=f"latest news about {city} Pakistan",
        max_results=2
    )

    results = response.get("results", [])

    if not results:
        return f"No latest news found for {city}."

    output = []

    for result in results:

        title = result.get("title", "No title")
        content = result.get("content", "No content")

        # Keep news content short
        content = content[:500]

        output.append(
            f"Title: {title}\n"
            f"Content: {content}"
        )

    return "\n\n".join(output)



# 6. CREATE GROQ LLM


llm = ChatGroq(
    model="openai/gpt-oss-120b"
)


# 7. TOOL APPROVAL MIDDLEWARE


@wrap_tool_call
def humanApproval(request, handler):

    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]

    print(f"\nAgent wants to call '{tool_name}'")
    print(f"Arguments: {tool_args}")

    approval = input("Approve? (yes/no): ")

    if approval.lower() != "yes":

        return ToolMessage(
            content="The user denied this tool call.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)


# 8. CREATE AGENT


agent = create_agent(
    llm,
    tools=[
        get_weather,
        search_city_news
    ],
    system_prompt=(
        "You are a helpful city assistant. "
        "Give short and direct answers when the user asks for short answers."
    ),
    middleware=[humanApproval]
)


# 9. CREATE RUNNABLE


def prepare_input(user_input):
    """Convert normal user input into agent message format."""

    return {
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    }


input_runnable = RunnableLambda(prepare_input)


# 10. RUN AGENT

print("============City Agent=============")
print("Type exit to quit")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Thank You!")
        break

    # Runnable converts user input into agent input
    agent_input = input_runnable.invoke(user_input)

    # Run the agent
    result = agent.invoke(agent_input)

    print("Bot:", result["messages"][-1].content)