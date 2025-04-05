import os
from dotenv import load_dotenv
import openai  # Import OpenAI library

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API client
api_key = os.environ.get("OPENAI_API_KEY", "")
openai.api_key = api_key  # Set OpenAI API key

def explain_automata(query: str) -> str:
    """
    Generate explanations about automata topics using OpenAI's API.
    
    Args:
        query: The automata-related query to explain
        
    Returns:
        A string explanation of the query
    """
    try:
        if not api_key:
            return "API key not configured. Please set the OPENAI_API_KEY environment variable."
        
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
        
        # Extract and return the explanation
        return response.choices[0].message.content
        
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
