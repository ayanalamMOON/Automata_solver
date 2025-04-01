import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

def explain_automata(query: str):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Explain {query} in simple terms"}]
    )
    return response["choices"][0]["message"]["content"]
