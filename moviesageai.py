# Moviesage is Ai bot that:

#1 takes a raw para about the movie
#2 Extract important structured info 
#3 Generate a clean summary 
#4  Returns it in a JSON format
#5 Stores it in there database 


from dotenv import load_dotenv
load_dotenv()

from typing import List, Optional
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI


model = ChatMistralAI(model="mistral-small-2603")


# Pydantic Schema
class Movie(BaseModel):
    title: str
    release_year: Optional[str] = None
    director: Optional[str] = None
    genre: List[str] = []
    main_cast: List[str] = []
    language: Optional[str] = None
    country: Optional[str] = None
    rating: Optional[str] = None
    runtime: Optional[str] = None
    plot_overview: str
    budget: Optional[str] = None
    box_office_collection: Optional[str] = None


# Parser
parser = PydanticOutputParser(
    pydantic_object=Movie
)


# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are MovieSage, an AI assistant specialized in analyzing movies.

Your task is to take raw, unstructured information about a movie and
transform it into structured JSON data.

Follow these rules:

1. Extract information only from the provided text.
2. Do not invent information.
3. If information is missing, use null or an empty list where appropriate.
4. Generate a clean and concise plot overview.
5. Return the result according to these formatting instructions:

{format_instructions}
"""
    ),
    (
        "human",
        """
Analyze the following raw movie information:

{movie_text}
"""
    )
])


para = input("Give your para about the movie: ")


final_prompt = prompt.invoke({
    "movie_text": para,
    "format_instructions": parser.get_format_instructions()
})


res = model.invoke(final_prompt)

# Convert model output into a Pydantic object
movie = parser.parse(res.content)

# JSON output
print(movie.model_dump_json(indent=2))
