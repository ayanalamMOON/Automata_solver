from flask import Flask, request, jsonify
from flask_cors import CORS
from ai_explainer import explain_automata, analyze_user_answer

app = Flask(__name__)
CORS(app)

@app.route('/api/explain', methods=['POST'])
def get_explanation():
    data = request.json
    query = data.get('query', '')
    explanation = explain_automata(query)
    return jsonify({'explanation': explanation})

@app.route('/api/analyze-answer', methods=['POST'])
def get_answer_analysis():
    data = request.json
    question = data.get('question', '')
    user_answer = data.get('answer', '')
    
    # Validate inputs
    if not question or not user_answer:
        return jsonify({'error': 'Both question and answer are required'}), 400
    
    analysis = analyze_user_answer(question, user_answer)
    return jsonify({'analysis': analysis})

if __name__ == '__main__':
    app.run(debug=True)