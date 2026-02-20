# Import required libraries
from flask import Flask, render_template, request   # Flask framework tools
from transformers import MarianMTModel, MarianTokenizer  # Translation model & tokenizer

# Create Flask application instance
app = Flask(__name__)

# -------------------------------
# Load Translation Model (English → Hindi)
# -------------------------------

# Model name from Hugging Face by Helsinki-NLP
model_name = "Helsinki-NLP/opus-mt-en-hi"

# Load tokenizer (converts text → tokens/numbers)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Load pre-trained translation model
model = MarianMTModel.from_pretrained(model_name)


# -------------------------------
# Function to Translate Text
# -------------------------------
def translate_text(text):
    
    # Convert input text into model-readable tokens (PyTorch format)
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    # Generate translated tokens using the model
    translated = model.generate(**inputs)
    
    # Convert translated tokens back to readable text
    output = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    # Return final translated text
    return output


# -------------------------------
# Define Home Route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    
    # Initialize empty translated text
    translated_text = ""
    
    # If user submits the form (POST request)
    if request.method == "POST":
        
        # Get text entered by user from HTML form
        text = request.form["text"]
        
        # Call translation function
        translated_text = translate_text(text)
    
    # Send translated result back to HTML page
    return render_template("index.html", translated_text=translated_text)


# -------------------------------
# Run Application
# -------------------------------
if __name__ == "__main__":
    
    # Start Flask development server
    # debug=True enables auto-reload & detailed error messages
    app.run(debug=True)
