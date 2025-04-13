import os
from dotenv import load_dotenv
import openai
from redis.connection import ConnectionPool
from redis.client import Redis
from redis.exceptions import ConnectionError, TimeoutError
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
import json
from datetime import timedelta
import logging
import functools
import time
from typing import Optional, Any

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Redis connection configuration with connection pooling
REDIS_CONFIG = {
    'host': os.environ.get('REDIS_HOST', 'localhost'),
    'port': int(os.environ.get('REDIS_PORT', 6379)),
    'db': 0,
    'decode_responses': True,
    'socket_timeout': 5,
    'retry_on_timeout': True
}

# Create a connection pool
redis_pool = ConnectionPool(**REDIS_CONFIG)

def get_redis_client() -> Redis:
    """Get Redis client with automatic retries and backoff"""
    retry = Retry(ExponentialBackoff(), 3)
    return Redis(connection_pool=redis_pool, retry=retry)

def with_redis_retry(max_retries: int = 3, backoff_factor: float = 0.1):
    """Decorator for Redis operations with retry logic"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError) as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Redis operation failed after {max_retries} attempts: {str(e)}")
                        raise
                    wait_time = backoff_factor * (2 ** attempt)
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

@with_redis_retry()
def get_cached_response(cache_key: str) -> Optional[str]:
    """Get response from cache if it exists with retry logic"""
    try:
        client = get_redis_client()
        return client.get(cache_key)
    except Exception as e:
        logger.error(f"Error getting cached response: {str(e)}")
        return None

@with_redis_retry()
def set_cached_response(cache_key: str, response: str, expire_time: int = 3600) -> None:
    """Store response in cache with expiration and retry logic"""
    try:
        client = get_redis_client()
        client.setex(cache_key, timedelta(seconds=expire_time), response)
    except Exception as e:
        logger.error(f"Error setting cached response: {str(e)}")

def with_openai_retry(max_retries: int = 3, backoff_factor: float = 0.1):
    """Decorator for OpenAI API calls with retry logic"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except openai.error.RateLimitError:
                    if attempt == max_retries - 1:
                        logger.error("OpenAI API rate limit exceeded")
                        raise
                    wait_time = backoff_factor * (2 ** attempt)
                    time.sleep(wait_time)
                except openai.error.APIError as e:
                    if attempt == max_retries - 1:
                        logger.error(f"OpenAI API error: {str(e)}")
                        raise
                    wait_time = backoff_factor * (2 ** attempt)
                    time.sleep(wait_time)
            return None
        return decorator

@with_openai_retry()
def explain_automata(query: str) -> str:
    """
    Generate explanations about automata topics using OpenAI's API with caching and retries.
    
    Args:
        query: The automata-related query to explain
        
    Returns:
        A string explanation of the query
    """
    try:
        if not openai.api_key:
            return "API key not configured. Please set the OPENAI_API_KEY environment variable."
        
        # Check cache first with hash of normalized query
        normalized_query = query.lower().strip()
        cache_key = f"automata_explanation:{hash(normalized_query)}"
        cached_response = get_cached_response(cache_key)
        if cached_response:
            logger.info("Cache hit for query: %s", normalized_query)
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
        
        # Cache the response with dynamic expiration based on query complexity
        expire_time = 7200 if len(query.split()) > 10 else 3600
        set_cached_response(cache_key, explanation, expire_time)
        
        return explanation
        
    except Exception as e:
        logger.error("Error generating explanation: %s", str(e))
        return f"Error generating explanation: {str(e)}"

@with_openai_retry()
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
        if not openai.api_key:
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
        logger.error("Error analyzing answer: %s", str(e))
        return f"Error analyzing answer: {str(e)}"
