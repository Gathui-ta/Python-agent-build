from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from tools import tools

load_dotenv()

class ResearchResponse(BaseModel):
   topic: str
   summary: str
   sources: list[str]
   tools_used: list[str]

llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
   [
      (
         "system",
         """
         You are a research assistant that will help generate a research paper.
         Answer the user query and use necessary tools.
         Wrap the output in this format and provide no other text\n{format_instructions}
         """,
      ),
      ("placeholder", "{chat_history}"),
      ("human", "{query}"),
      ("placeholder", "{agent_scratchpad}"),
   ]

).partial(format_instructions=parser.get_format_instructions())

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a research assistant. Answer the user's research query.",
    response_format=ResearchResponse,
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the capital city of france?"}
    ]
})
print(result["structured_response"])