import os
from dotenv import load_dotenv
from gemini import GeminiClient

# Load environment variables from .env file
load_dotenv()

# Initialize Gemini API client
api_key = os.environ.get("GEMINI_API_KEY", "")
gemini_client = GeminiClient(api_key=api_key)

def explain_automata(query: str) -> str:
    """
    Generate explanations about automata topics using Gemini's API.
    
    Args:
        query: The automata-related query to explain
        
    Returns:
        A string explanation of the query
    """
    try:
        if not api_key:
            return "API key not configured. Please set the GEMINI_API_KEY environment variable."
        
        # Prepare context for better automata explanations
        context = """
        You are an expert in automata theory, formal languages, and computational theory.
        Provide clear, concise explanations about the following automata-related query.
        Include relevant examples where appropriate.
        """
        
        # Call the Gemini API
        response = gemini_client.chat.completions.create(
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
