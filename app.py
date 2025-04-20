from flask import Flask, request, jsonify, render_template
from chatbot import *

app = Flask(__name__)

print("Initializing chatbot...")
initialize_chatbot()
print("Chatbot is ready!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message") 
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    response = process_input(user_input)
    return jsonify({"response": response}) 


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)