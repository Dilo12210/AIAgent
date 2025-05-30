from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor


# loads env variable file from .env file
load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    reccommendations: list[str]

llm = ChatOpenAI(
    model="gpt-4o-mini",

)

parser = PydanticOutputParser(pydantic_object=ResearchResponse) # can use json but pydantic is popular choice
prompt = ChatPromptTemplate.from_messages(
    [
    ("system", "You are an agent for a commerce website. You should reccommend products to the" 
     "user based on their needs. Answer the questions in a concise and friendly manner to hold general" 
     "conversation and always follow up if they need any other help. Wrap the output in this format and provide no other text\n{format_instructions}"),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
]
).partial(format_instructions=parser.get_format_instructions()) # format instructions are used to format the output

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[]
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[],
    verbose=True,
)
while True:
    # chat_history = "You are a helpful assistant. How can I help you today?"
    # query = input("What would you like to purchase? ")
    # raw_response = agent_executor.invoke({"query": query, "chat_history": chat_history})
    # print(raw_response)
    # chat_history += f"\nUser: {query}\nAssistant: {raw_response.get('output')}"
    
    # For the sake of this example, we will not use chat history
    query = input("What would you like to purchase? ")
    raw_response = agent_executor.invoke({"query": query})

    try:
        structured_response = parser.parse(raw_response.get("output")[0]["text"])
        print(structured_response + "\nWhat else can I help you with?")
    except Exception as e:
        print("Error parsing response:", e)