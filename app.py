from flask import Flask, render_template, request, jsonify
from transformers import MarianMTModel, MarianTokenizer

# Create Flask application instance
app = Flask(__name__)

# Load translation model once (fast after first load)
model_name = "Helsinki-NLP/opus-mt-en-hi"

# Load tokenizer (converts text → tokens/numbers)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Load pre-trained translation model
model = MarianMTModel.from_pretrained(model_name)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data["text"]

    # Tokenize input
    tokens = tokenizer(text, return_tensors="pt", padding=True)

    # Generate translation
    translated = model.generate(**tokens)

    # Decode translation
    output = tokenizer.decode(translated[0], skip_special_tokens=True)

    return jsonify({"translation": output})

if __name__ == "__main__":
    
    # Start Flask development server
    # debug=True enables auto-reload & detailed error messages
    app.run(debug=True)
