from flask import Flask, request
from chatbot.chatbot import predict, get_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/api/chatbot", methods=['POST'])
def hello_world():
    data = request.get_json()

    if not 'message' in data:
        return {"error": "Please provide a message."}

  
    message = data['message']
    try:
        pred = predict(message)
        bot_response = get_response(pred)

        if pred == 'name':
            bot_response = f"{bot_response} this website" 

        return {"message": bot_response}
    except Exception as e:
        return {"error": str(e)}
 

if __name__ == "__main__":
    app.run()


