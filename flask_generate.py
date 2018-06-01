from flask import Flask
from flask import request
from flask import render_template
from generate_char import generate

app = Flask(__name__)

@app.route('/')
def my_form():
    return(render_template('user_input.html'))

@app.route('/', methods=['POST'])
def my_form_post():
    user_input = request.form['text']
    return(generate(user_input))