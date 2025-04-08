import os
from dotenv import load_dotenv
import openai
import redis
import json
from datetime import timedelta

# Load environment variables
load_dotenv()

# Initialize Redis client
redis_client = redis.Redis(
    host=os.environ.get('REDIS_HOST', 'localhost'),
    port=int(os.environ.get('REDIS_PORT', 6379)),
    db=0,
    decode_responses=True
)

# Initialize OpenAI API client
api_key = os.environ.get("OPENAI_API_KEY", "")
openai.api_key = api_key

def get_cached_response(cache_key: str) -> str:
    """Get response from cache if it exists"""
    return redis_client.get(cache_key)

def set_cached_response(cache_key: str, response: str, expire_time: int = 3600) -> None:
    """Store response in cache with expiration"""
    redis_client.setex(cache_key, timedelta(seconds=expire_time), response)

def explain_automata(query: str) -> str:
    """
    Generate explanations about automata topics using OpenAI's API with caching.
    
    Args:
        query: The automata-related query to explain
        
    Returns:
        A string explanation of the query
    """
    try:
        if not api_key:
            return "API key not configured. Please set the OPENAI_API_KEY environment variable."
        
        # Check cache first
        cache_key = f"automata_explanation:{hash(query)}"
        cached_response = get_cached_response(cache_key)
        if cached_response:
            return cached_response

        # Prepare context for better automata explanations
        context = """
        You are an expert in automata theory, formal languages, and computational theory.
        Provide clear, concise explanations about the following automata-related query.
        Include relevant examples where appropriate.
        """
        
        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": query}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        explanation = response.choices[0].message.content
        # Cache the response
        set_cached_response(cache_key, explanation)
        return explanation
        
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def analyze_user_answer(question: str, user_answer: str) -> str:
    """
    Analyze a user's answer to an automata question and provide feedback.
    
    Args:
        question: The original automata question/problem
        user_answer: The user's answer to be analyzed
        
    Returns:
        A string with feedback and suggestions for improvement
    """
    try:
        if not api_key:
            return "API key not configured. Please set the OPENAI_API_KEY environment variable."
        
        # Prepare context for analyzing automata answers
        context = """
        You are an expert in automata theory, formal languages, and computational theory.
        Analyze the user's answer to the given automata problem.
        Provide constructive feedback by:
        1. Identifying correct aspects of their solution
        2. Pointing out any errors or misconceptions
        3. Suggesting specific improvements
        4. Offering additional insights or alternative approaches when relevant
        Be educational and supportive in your feedback.
        """
        
        # Construct the prompt with question and answer
        prompt = f"Question: {question}\n\nUser's Answer: {user_answer}"
        
        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.5
        )
        
        # Extract and return the analysis
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error analyzing answer: {str(e)}"
