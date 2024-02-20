

import pandas as pd
import weaviate
import weaviate.classes as wvc
import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()
import openai
import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, BootstrapFinetune
from dspy.retrieve.weaviate_rm import WeaviateRM
openai.api_key = os.getenv("OPENAI_API_KEY")
weaviate_key = os.getenv("WEAVIATE_API_KEY")

# Define the URL for the Weaviate cluster
WCS_CLUSTER_URL = "https://bookstore-copilot-w6cbjbif.weaviate.network"

# Connect to the Weaviate cluster using the URL and API key
client = weaviate.connect_to_wcs(
    cluster_url=WCS_CLUSTER_URL,  # URL of the Weaviate cluster
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),  # Authentication using API key
    headers={
        "X-OpenAI-Api-Key": openai.api_key  # Header for OpenAI API key
    },
    skip_init_checks=True,  # Skip initial checks for faster setup
)

# Retrieve the 'book_summaries' collection from the Weaviate client
questions = client.collections.get("book_summaries")

# Initialize the DSPy library with OpenAI's GPT-3.5-turbo model
llm = dspy.OpenAI(model="gpt-3.5-turbo")
# Configure DSPy settings to use the initialized language model
dspy.settings.configure(lm=llm)

# Retrieve the 'book_summaries' collection again for further operations
book_collection = client.collections.get('book_summaries')

class FunctionCalling(dspy.Signature):
    """" There are 5 functions and their function signature.
    1. Book search by title get_books_by_title(query: str) -> List[str]
    2. Book seach by author get_books_by_author(query: str) -> List[str]
    3. Book seach by genre get_books_by_genre(query: str) -> List[str]
    4. Book recommendations based on user preferences get_recommendations(query: str) -> List[str]
    5. More about the book get_more_about_the_book(title: str) -> str
    5. General enquires about shipping, policies and returns get_general_enquires(query: str) -> str
    Based on the user input, the module will call the appropriate function with the required parameters inferred from the query and return the response.
    """
    query = dspy.InputField(desc='Input query')
    function_name = dspy.OutputField(desc='Function name to be called along with the parameters')

class BookInterestFromQuery(dspy.Signature):
    """This module will take the user query and infer a string which has all necessary attributes that the
    user wants to search for.
    For e.g. I want a book on magic and adventure -> infer 'magic adventure' as the user preferences
    """
    query = dspy.InputField(desc='Input query')
    user_preferences = dspy.OutputField(desc='Inferred user preferences')

class GenreFromQuery(dspy.Signature):
    """This module will take the user query and infer the genre of the book that the user is interested in.
    For e.g. I want a book on magic and adventure -> infer 'fantasy' as the genre
    """
    query = dspy.InputField(desc='Input query')
    genre = dspy.OutputField(desc='Inferred genre')

class AuthorFromQuery(dspy.Signature):
    """This module will take the user query and infer the author of the book that the user is interested in.
    For e.g. I want a book by J.K. Rowling -> infer 'J.K. Rowling' as the author
    """
    query = dspy.InputField(desc='Input query')
    author = dspy.OutputField(desc='Inferred author')

class TitleFromQuery(dspy.Signature):
    """This module will take the user query and infer the title of the book that the user is interested in.
    For e.g. I want a book with the title Harry Potter -> infer 'Harry Potter' as the title
    """
    query = dspy.InputField(desc='Input query')
    title = dspy.OutputField(desc='Inferred title')

class GeneralEnquiresFromQuery(dspy.Signature):
    """This module will take the user query and infer the general enquires that the user is interested in.
    There are only 3 categories possible, shipping, policies and returns.
    For e.g. I want to know about shipping -> infer 'shipping' as the general enquires
    """
    query = dspy.InputField(desc='Input query')
    general_enquires = dspy.OutputField(desc='Inferred general enquires')

class EnquiryAnswerGenerator(dspy.Signature):
    """Answer general queries by the customer. General queries can be related to shipping, returns and policies.
    You are Amazon's virtual assistant and you have to answer the general queries of the customer.
    There are only 3 categories possible, shipping, policies and returns.
    """
    query = dspy.InputField(desc='General enquires')
    response = dspy.OutputField(desc='Response to the general enquires')

class MoreAboutTheBook(dspy.Signature):
    """Given the title of the book, the module will return the summary of the book.
    """
    title = dspy.InputField(desc='Title of the book')
    summary = dspy.OutputField(desc='Summary of the book')

# Initialize ChainOfThought processes for various query types with n=2 for deeper analysis
book_interest_from_query = dspy.ChainOfThought(BookInterestFromQuery, n=2)
title_from_query = dspy.ChainOfThought(TitleFromQuery, n=2)
genre_from_query = dspy.ChainOfThought(GenreFromQuery, n=2)
author_from_query = dspy.ChainOfThought(AuthorFromQuery, n=2)
general_enquires_from_query = dspy.ChainOfThought(GeneralEnquiresFromQuery, n=2)
enquiry_answer_generator = dspy.ChainOfThought(EnquiryAnswerGenerator, n=2)
function_calling = dspy.ChainOfThought(FunctionCalling, n=2)
more_about_the_book = dspy.ChainOfThought(MoreAboutTheBook, n=2)

# Function to get books by title using the title inferred from the user's query
def get_books_by_title(query: str) -> str:
    title = book_interest_from_query(query=query).title
    response = book_collection.query.near_text(
        query=title,
        limit=4
    )
    # Format the response to include both title and author
    res = [obj.properties['title'] + ' by ' + obj.properties['author'] for obj in response.objects]
    return res

# Function to get books by author using the author inferred from the user's query
def get_books_by_author(query: str) -> str:
    author = author_from_query(query=query).author
    response = book_collection.query.near_text(
        query=author,
        limit=4
    )
    # Format the response to include both title and author
    res = [obj.properties['title'] + ' by ' + obj.properties['author'] for obj in response.objects]
    return res

# Function to get books by genre using the genre inferred from the user's query
def get_books_by_genre(query: str) -> str:
    genre = genre_from_query(query=query).genre
    response = book_collection.query.near_text(
        query=genre,
        limit=4
    )
    # Format the response to include both title and author
    res = [obj.properties['title'] + ' by ' + obj.properties['author'] for obj in response.objects]
    return res

# Function to get book recommendations based on user preferences inferred from the query
def get_recommendations(query: str) -> str:
    user_preferences = book_interest_from_query(query=query).user_preferences
    response = book_collection.query.near_text(
        query=user_preferences,
        limit=4
    )
    # Format the response to include both title and author
    res = [obj.properties['title'] + ' by ' + obj.properties['author'] for obj in response.objects]
    return res

# Function to get answers to general enquiries using the response inferred from the user's query
def get_general_enquires(query: str) -> str:
    general_enquires = enquiry_answer_generator(query=query).response
    return general_enquires

# Function to get more details about a book using the title inferred from the user's query
def get_more_about_the_book(title: str) -> str:
    title = title_from_query(query=title).title
    response = book_collection.query.near_text(
        query=title,
        limit=1
    )
    # Extract the summary from the response
    summary = response.objects[0].properties['book_summaries']
    return summary


class ResponseCrafter(dspy.Signature):
    """ You will given some data and a query from the user. You need to craft a polite response, for the query, using the data provided.
    """
    response = dspy.InputField(desc='Response')
    query = dspy.InputField(desc='User query')
    crafted_response = dspy.OutputField(desc='Crafted response')


response_crafter = dspy.ChainOfThought(ResponseCrafter, n=2)
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

while 1:
    try:
        query = input(Fore.YELLOW + Style.BRIGHT + "Enter your query: ")
        if query == 'exit':
            break
        function_name = function_calling(query=query).function_name
        print(Fore.GREEN + Style.BRIGHT + function_name)
        res = eval(function_name)
        if isinstance(res, list):
            res = '\n'.join(res)
        response = response_crafter(response=res, query=query).crafted_response
        print(Fore.GREEN + Style.BRIGHT + response)
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + "Sorry, I didn't understand that. Please try again.")


