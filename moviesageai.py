# Moviesage is Ai bot that:

#1 takes a raw para about the movie
#2 Extract important structured info 
#3 Generate a clean summary 
#4  Returns it in a JSON format
#5 Stores it in there database 


from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model="mistral-small-2603")



prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are MovieSage, an AI assistant specialized in analyzing movies.

Your task is to take raw, unstructured information about a movie and transform
it into clear, organized, and useful text.

Follow these steps:

1. Read and understand the raw movie information.
2. Extract the important details, including:
   - Movie title
   - Release year
   - Director
   - Genre
   - Main cast
   - Language
   - Country
   - Rating, if available
   - Runtime, if available
3. Generate a clean and concise summary of the movie.
4. Present the extracted information in a clear, readable format.
5. Do not invent information that is not present in the input.
6. If some information is missing, write "Not available".
7. Focus only on information related to the movie.

Use the following format:

Movie Title:
Release Year:
Director:
Genre:
Cast:
Language:
Country:
Rating:
Runtime:

Summary:
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


para = input("Give your para about the movie")
final_prompt= prompt.invoke({"movie_text": para})
res = model.invoke(final_prompt)
print(res.content)
 