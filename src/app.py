from flask import Flask, request, jsonify
import uuid
import os
from rag import rag_llm
import db


app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing question"}), 400

        question = data['question']
        
        # Run RAG pipeline
        answer_data = rag_llm(question)
        # Generate a unique conversation ID
        conversation_id = str(uuid.uuid4())
        
        db.save_conversation(
            conversation_id=conversation_id,
            question=question,
            answer_data=answer_data,
        )

        return jsonify({
            "conversation_id": conversation_id,
            "question": question,
            "answer": answer_data["answer"]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    if not data or 'conversation_id' not in data or 'feedback' not in data:
        return jsonify({"error": "Missing conversation_id or feedback"}), 400

    conversation_id = data['conversation_id']
    feedback_value = data['feedback']  # expected: +1 or -1
    
    if feedback_value not in [1, -1]:
        return jsonify({"error": "Feedback must be +1 or -1"}), 400
    
    db.save_feedback(
        conversation_id=conversation_id,
        feedback=feedback_value,
    )
    
    result = {
        "message": f"Feedback received for conversation {conversation_id}: {feedback_value}"
    }
    return jsonify(result)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    app.run(debug=True, port=5001)
