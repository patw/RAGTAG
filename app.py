from flask import Flask, render_template, redirect, url_for, request, session
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, SelectField, PasswordField, IntegerField, FloatField
from wtforms.validators import DataRequired
import os
import json
import pymongo
from bson import ObjectId
from datetime import datetime, timedelta
from dotenv import load_dotenv
import functools

# Instructor-large embedding model
from InstructorEmbedding import INSTRUCTOR
instructor_model = INSTRUCTOR('hkunlp/instructor-large')

# Get environment variables
load_dotenv()

# Create the Flask app object
app = Flask(__name__)

# Need this for storing anything in session object
app.config['SECRET_KEY'] = os.environ["SECRET_KEY"]

# Load users from .env file
users_string = os.environ["USERS"]
users = json.loads(users_string)

# Connect to mongo
conn = os.environ["MONGO_CON"]
database = os.environ["MONGO_DB"]
collection = os.environ["MONGO_COL"]
client = pymongo.MongoClient(conn)
db = client[database]
col = db[collection]

# Make it pretty because I can't :(
Bootstrap(app)

# Flask forms is magic
class ChunkForm(FlaskForm):
    chunk_question = StringField('Question', validators=[DataRequired()])
    chunk_answer = TextAreaField('Answer', validators=[DataRequired()])
    chunk_enabled = SelectField('Enabled', choices=[(True, 'Enabled'), (False, 'Disabled')])
    submit = SubmitField('Submit')

# Amazing, I hate writing this stuff
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# Always have a search bar
class SearchForm(FlaskForm):
    search_string = StringField('Question/Answer Search', validators=[DataRequired()])
    submit = SubmitField('Submit')

# Always have a search bar - for vectors too
class VectorSearchForm(FlaskForm):
    search_string = StringField('Vector Search', validators=[DataRequired()])
    search_k = IntegerField("K Value", validators=[DataRequired()])
    search_score_cut = FloatField("Score Cut Off", validators=[DataRequired()])
    submit = SubmitField('Submit')

# Return embedding with instruction and text
def get_embedding(ins, text):
    return instructor_model.encode([[ins,text]]).tolist()[0]

# Atlas search query for chunks
def search_chunks(search_string):
    search_query = [
        {
            "$search": {
                "text": {
                    "path": ["chunk_question", "chunk_answer"],
                    "query": search_string
                }
            }
        },
        {
            "$limit": 25
        },
        {
            "$project": {
                "_id": 1,
                "chunk_question": 1,
                "chunk_answer": 1,
                "chunk_enabled": 1,
                "score": {"$meta": "searchScore"}
            }
        }]

    return col.aggregate(search_query)

def test_chunks(search_string, k, cut):
    v = get_embedding("Represent the question for retrieving supporting documents:", search_string)
    search_query = [
        {
            "$search": {
                "knnBeta": {
                    "path": "chunk_embedding",
                    "vector": v,
                    "k": int(k)
                }
            }
        },
        {
            "$limit": 5
        },
        {
            "$project": {
                "_id": 1,
                "chunk_question": 1,
                "chunk_answer": 1,
                "chunk_enabled": 1,
                "score": {"$meta": "searchScore"}
            }
        },
        {
            "$match": { "score": { "$gte": float(cut) }}
        }
        ]
    
    print(search_query)

    return col.aggregate(search_query)


# Define a decorator to check if the user is authenticated
# No idea how this works...
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if session.get("user") is None:
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view

# The default chunk view, ordered by priority and highlighted in red if overdue
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():

    # We're doing a search here
    form = SearchForm()
    if request.method == "POST":
        form_result = request.form.to_dict(flat=True)
        chunks = search_chunks(form_result["search_string"])
        return render_template('search.html', chunks=chunks)

    # Get the chunks!
    chunk_query = col.find().skip(0).limit(50)

    # Get all the active chunks
    chunks = []
    for chunk_item in chunk_query:
        chunks.append(chunk_item)

    # Spit out the template
    return render_template('index.html', chunks=chunks, form=form)

@app.route('/test', methods=['GET', 'POST'])
@login_required
def test():

    # no chunks by default
    chunks = []

    # We're doing a vector search here
    form = VectorSearchForm(search_k=100, search_score_cut=0.88)
    if request.method == "POST":
        form_result = request.form.to_dict(flat=True)
        chunks = test_chunks(form_result["search_string"], form_result["search_k"], form_result["search_score_cut"])
        return render_template('test.html', chunks=chunks, form=form)

    # Spit out the template
    return render_template('test.html', chunks=chunks, form=form)

# Create or edit chunks
@app.route('/chunk', methods=['GET', 'POST'])
@app.route('/chunk/<id>', methods=['GET', 'POST'])
@login_required
def chunk(id=None):
    form = ChunkForm()
    if request.method == "POST":
        # Get the form result back and clean up the data set
        form_result = request.form.to_dict(flat=True)
        form_result.pop('csrf_token')
        form_result.pop('submit')
        embed_text = form_result["chunk_question"] + " " + form_result["chunk_answer"]
        print(embed_text)
        form_result["chunk_embedding"] = get_embedding("Represent the document for retrieval:", embed_text)

        # Store the result in mongo collection
        if id:
            col.replace_one({'_id': ObjectId(id)}, form_result)
        else:
            col.insert_one(form_result)
            # Back to the chunk view
        return redirect("/")
    else:
        if id:
            chunk = col.find_one({'_id': ObjectId(id)})
            form.chunk_question.data = chunk["chunk_question"]
            form.chunk_answer.data = chunk["chunk_answer"]
            form.chunk_enabled.data = chunk["chunk_enabled"]
    return render_template('chunk.html', form=form)

# This chunk is bad, we need to make it feel bad
@app.route('/chunk_disable/<id>')
@login_required
def chunk_disable(id):
    update_doc = {
        "chunk_enabled": False
    }

    chunk_data = col.find_one({'_id': ObjectId(id)})
    col.update_one({'_id': ObjectId(id)}, {"$set": update_doc})
    return redirect('/')

# Login/logout routes that rely on the user being stored in session
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.username.data in users:
            if form.password.data == users[form.username.data]:
                session["user"] = form.username.data
                return redirect(url_for('index'))
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    session["user"] = None
    return redirect(url_for('login'))