from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer  # Translation model & tokenizer

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    translated_msg = ""
    if request.method == 'POST':
        data = request.form['message']
        translated_msg = translation(data)
    return render_template('index.html', output=translated_msg)

if __name__ == '__main__':
        app.run(debug=True)     
    
 