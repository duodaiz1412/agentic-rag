from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda

from graph.chains.llm_config import create_llm, rate_limit_delay

load_dotenv()


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Route the user query to the vectorstore or websearch. Avalable options are 'vectorstore' or 'web_search'",
    )


llm = create_llm(model="gemini-2.5-flash", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

message = """You are an expert router that decides whether a user's question should be answered using the internal vectorstore or web search.

The vectorstore contains educational content from an e-learning platform including:
- Course overviews and descriptions
- Lesson content (text, video transcripts, attachments)
- Audio transcripts from video lessons (translated to English when available)
- Topics covering: programming, databases (PostgreSQL, MySQL, etc.), web development, software engineering, and various technical subjects

Routing rules:

1. Route to **vectorstore** for questions about:
   - Course content, lessons, transcripts, or educational materials
   - Technical concepts, definitions, explanations (programming, SQL, frameworks, algorithms, etc.)
   - How-to guidance, code examples, tutorials, or walkthroughs that could be in course materials
   - Any educational/technical content that might be covered in courses

2. Route to **web_search** only when:
   - User explicitly asks for current/time-sensitive info (news, latest releases, version numbers, CVEs, "today", "2025", etc.)
   - Question is clearly outside course scope (general news, politics, sports, weather, real-time data, etc.)
   - User explicitly requests external sources or citations

3. When uncertain:
   - Default to **vectorstore** (it contains comprehensive educational content)
   - Only use web_search if question clearly requires real-time or external information

Important:
- Prefer vectorstore when in doubt - it has extensive educational content
- The vectorstore contains English content; route to vectorstore if question is in English or about technical topics
- Be conservative: only route to web_search when absolutely necessary for current events or external sources
"""
router_prompt = ChatPromptTemplate.from_messages(
    [("system", message), ("human", "{question}")]
)

base_router = router_prompt | structured_llm_router


def _rate_limited_invoke(input_dict: dict):
    """Wrapper to add rate limiting to router chain"""
    rate_limit_delay()
    return base_router.invoke(input_dict)


question_router = RunnableLambda(_rate_limited_invoke)
