# Basic flask stuff for building http APIs and rendering html templates
from flask import Flask, render_template, redirect, url_for, request, session

# Bootstrap integration with flask so we can make pretty pages
from flask_bootstrap import Bootstrap

# Flask forms integrations which save insane amounts of time
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, SelectField, PasswordField, IntegerField, FloatField
from wtforms.validators import DataRequired

# Basic python stuff
import os
import json
import functools

# Basic mongo python stuff
import pymongo
from bson import ObjectId
from bson import json_util

# Nice way to load environment variables for deployments
from dotenv import load_dotenv

# Instructor-large embedding model for creating vectors
from InstructorEmbedding import INSTRUCTOR
instructor_model = INSTRUCTOR('hkunlp/instructor-large')

# Use the wonderful llama.cpp library to execute our LLM (mistral-7b with dolphin fine tune)
from llama_cpp import Llama
llama_model = Llama(model_path="dolphin-2.1-mistral-7b.Q5_K_S.gguf")
system_message = "You are a helpful assistant who will always answer the question with only the data provided and in 3 sentences."
prompt_format = "<|im_start|>system\n" + system_message + "<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant:"
ban_token = "<|im_end|>"  # This is to prevent the model from leaking additional questions

# Get environment variables
load_dotenv()

# Create the Flask app object
app = Flask(__name__)

# Load API key from .evn file - super secure
api_key = os.environ["API_KEY"]

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

# Vector search but now for the chatbot LLM
class LLMForm(FlaskForm):
    question = StringField('Question', validators=[DataRequired()])
    search_k = IntegerField("K Value", validators=[DataRequired()])
    search_score_cut = FloatField("Score Cut Off", validators=[DataRequired()])
    llm_prompt = TextAreaField('Prompt', validators=[DataRequired()])
    llm_tokens = IntegerField("Number of tokens from LLM", validators=[DataRequired()])
    submit = SubmitField('Submit')

# Return embedding with instruction and text
def get_embedding(ins, text):
    return instructor_model.encode([[ins,text]]).tolist()[0]

# Return the retrieval augmented generative result
def get_rag(question, search_k, search_score_cut, llm_prompt, llm_tokens):
    # Get all the chunks
    chunks = list(vector_search_chunks(question, search_k, search_score_cut))

    # Build the LLM answer chunks
    answers = ""
    for answer in chunks:
        answers = answers + answer["chunk_answer"] + " "

    # Replace the template tokens with the question and the answers
    prompt = llm_prompt.replace("%q%", question)
    prompt = prompt.replace("%d%", answers)

    # One more replacement step to help our chat model out with a system prompt and proper control tokens
    llm_result = {}
    llm_result["input"] = prompt_format.replace("{prompt}", prompt)

    # Generate LLM response and return the text
    llm_result["output"] = llama_model(llm_result["input"], max_tokens=llm_tokens, temperature=0.1)["choices"][0]["text"]

    # Find the baned tokens
    index = llm_result["output"].find(ban_token)

    # Check if the ban token is found in the string
    if index != -1:
        # Trim the string, including the marker and everything after it
        llm_result["output"] = llm_result["output"][:index]

    return llm_result

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

# Altlas vector search query for testing chunks semantically using embeddings
def vector_search_chunks(search_string, k, cut):
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

    return col.aggregate(search_query)


# Define a decorator to check if the user is authenticated
# No idea how this works...  Magic.
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        # Load users from .env file - this is sketchy security
        if "USERS" in os.environ:
            users_string = os.environ["USERS"].strip()
            users = json.loads(users_string)
            if session.get("user") is None:
                return redirect(url_for('login'))
        return view(**kwargs)        
    return wrapped_view

# The default chunk view with pagination and lexical search
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():

    # We're doing a lexical search here
    form = SearchForm()
    if request.method == "POST":
        form_result = request.form.to_dict(flat=True)
        chunks = search_chunks(form_result["search_string"])
        return render_template('search.html', chunks=chunks)

    # Get the chunks!
    chunk_query = col.find().skip(0).limit(50)
    chunks = []
    for chunk_item in chunk_query:
        chunks.append(chunk_item)

    # Spit out the template
    return render_template('index.html', chunks=chunks, form=form)

# We use this for doing semantic search testing on the chunks
@app.route('/test', methods=['GET', 'POST'])
@login_required
def test():

    # no chunks by default
    chunks = []

    # We're doing a vector search here
    form = VectorSearchForm(search_k=100, search_score_cut=0.89)
    if request.method == "POST":
        form_result = request.form.to_dict(flat=True)
        chunks = vector_search_chunks(form_result["search_string"], form_result["search_k"], form_result["search_score_cut"])
        return render_template('test.html', chunks=chunks, form=form)

    # Spit out the template
    return render_template('test.html', chunks=chunks, form=form)

# We use this for doing semantic search testing on the chunks
@app.route('/llm', methods=['GET', 'POST'])
@login_required
def llm():
    # no chunks by default
    chunks = []

    # We're doing a vector search here
    form = LLMForm(search_k=100, search_score_cut=0.89, llm_prompt="Answer the following question \"%q%\" using only this data while ignoring any data irrelevant to this question: %d%", llm_tokens=128)
    if request.method == "POST":
        form_result = request.form.to_dict(flat=True)
        llm_response = get_rag(form_result["question"], form_result["search_k"], form_result["search_score_cut"], form_result["llm_prompt"], int(form_result["llm_tokens"]))
        return render_template('llm.html', chunks=chunks, form=form, llm_response=llm_response["output"],prompt=llm_response["input"])

    # Spit out the template
    return render_template('llm.html', chunks=chunks, form=form)

# Create or edit chunks. Basic CRUD functionality.
@app.route('/chunk', methods=['GET', 'POST'])
@app.route('/chunk/<id>', methods=['GET', 'POST'])
@login_required
def chunk(id=None):

    # This is the input form we want to load for doing chunk add/edit
    form = ChunkForm()

    # POST means we're getting a completed form
    if request.method == "POST":
        # Get the form result back and clean up the data set
        form_result = request.form.to_dict(flat=True)
        form_result.pop('csrf_token')
        form_result.pop('submit')
        embed_text = form_result["chunk_question"] + " " + form_result["chunk_answer"]
        form_result["chunk_embedding"] = get_embedding("Represent the document for retrieval:", embed_text)

        # Store the result in mongo collection
        if id:
            col.replace_one({'_id': ObjectId(id)}, form_result)
        else:
            col.insert_one(form_result)
            # Back to the chunk view
        return redirect("/")
    else:
        # This is if we got passed a mongo document ID and we need to edit it.
        # Load the doc up and render the edit form.
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

# We finally have a link for this now!
@app.route('/logout')
def logout():
    session["user"] = None
    return redirect(url_for('login'))

# API endpoint for sending a question and getting the LLM output (RAG)
# This is what you want to call from your website, slack or discord bot.
@app.route('/api/rag')
def api_rag():
    key = request.args.get("key")
    q = request.args.get("q")
    
    # Make sure we have a valid key and question
    if not key:
        return {'error': 'no API key provided - /api/rag/key=<api key>'}
    if not q:
        return {'error': 'No q parameter found. You must ask a question - /api/rag/q=<string>'}
    if key != api_key:
        return {'error': 'API key does not match'}
    
    # Get the LLM result for the query
    return get_rag(q, 100, 0.89, "Answer the following question \"%q%\" using only this data while ignoring any data irrelevant to this question: %d%", 128)

# API endpoint for sending a question and getting the LLM output (RAG)
# This is what you want to call from your website, slack or discord bot.
@app.route('/api/vector')
def api_vector():
    key = request.args.get("key")
    q = request.args.get("q")
    
    # Make sure we have a valid key and question
    if not key:
        return {'error': 'no API key provided - /api/vector/key=<api key>'}
    if not q:
        return {'error': 'No q parameter found. You must provide a string to vectorize - /api/vector/q=<string>'}
    if key != api_key:
        return {'error': 'API key does not match'}
    
    # Get the vector result for the string
    return get_embedding("Represent the question for retrieving supporting documents:", q)

# API Endpoint to dump all stored chunks
@app.route('/api/list')
def api_list():
    key = request.args.get("key")
    # Make sure we have a valid key and question
    if not key:
        return {'error': 'no API key provided - /api/list/key=<api key>'}
    if key != api_key:
        return {'error': 'API key does not match'}

    # Get the chunks!
    chunk_query = col.find().skip(0).limit(50)
    chunks = []
    for chunk_item in chunk_query:
        chunks.append(chunk_item)
    return json.loads(json_util.dumps(chunks))

# API Endpoint to perform lexical search
@app.route('/api/search')
def api_search():
    key = request.args.get("key")
    q = request.args.get("q")
    
    # Make sure we have a valid key and question
    if not key:
        return {'error': 'no API key provided - /api/search/key=<api key>'}
    if not q:
        return {'error': 'No q parameter found. You must ask a question - /api/search/q=<string>'}
    if key != api_key:
        return {'error': 'API key does not match'}
    
    chunks = search_chunks(q)
    return json.loads(json_util.dumps(chunks))

# API Endpoint to perform vector search
@app.route('/api/vector_search')
def api_vector_search():
    key = request.args.get("key")
    q = request.args.get("q")
    
    # Make sure we have a valid key and question
    if not key:
        return {'error': 'no API key provided - /api/search/key=<api key>'}
    if not q:
        return {'error': 'No q parameter found. You must ask a question - /api/search/q=<string>'}
    if key != api_key:
        return {'error': 'API key does not match'}
    
    chunks = vector_search_chunks(q, 100, 0.89)
    return json.loads(json_util.dumps(chunks))
