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
import random

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
llama_model = Llama(model_path="dolphin-2.1-mistral-7b.Q5_K_S.gguf", n_ctx=2048, use_mlock=False)
prompt_format = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant:"
ban_token = "<|"  # This is to prevent the model from leaking additional questions

# Some default constants, feel free to change any of these!
# ----------------------------------------------------------
DEFAULT_SCORE_CUT = 0.89  # The score cut off for instructor results anywhere from 0.8 to 0.92 seems good
DEFAULT_TEMP = 0.1 # The LLM temperature value, 0.1 is deterministic results, 0.7 is more creative
DEFAULT_K = 100  # The over-request value for the ANN query. 100-200 is good.
DEFAULT_TOKENS = 64  # The default number of tokens for the LLM to produce.  64 is fast, 128 gives longer results.
# This is the default prompt with replaceable question (%q%) and data (%d%) tokens
DEFAULT_PROMPT = "Answer the following question \"%q%\" using only this data while ignoring any data irrelevant to this question: %d%"
# This is the default system message for controlling the LLM behavior
DEFAULT_SYSTEM = "You are a helpful assistant who will always answer the question with only the data provided and in 2 sentences"
# ----------------------------------------------------------

# Get environment variables
load_dotenv()

# Create the Flask app object
app = Flask(__name__)

# Load API key from .evn file - super secure
if "API_KEY" in os.environ:
    api_key = os.environ["API_KEY"]
else:
    api_key = None

# Need this for storing anything in session object
if "SECRET_KEY"  in os.environ:
    app.config['SECRET_KEY'] = os.environ["SECRET_KEY"].strip()
else:
    app.config['SECRET_KEY'] = "ohboyyoureallyshouldachangedthis"

# Connect to mongo using our loaded environment variables from the .env file
if "SPECUIMDBCONNSTR"  in os.environ:
    conn = os.environ["SPECUIMDBCONNSTR"].strip()
else:
    conn = os.environ["MONGO_CON"].strip()

if "MONGO_DB" in os.environ:
    database = os.environ["MONGO_DB"].strip()
else:
    database = "specialists"

if "MONGO_COL" in os.environ:
    collection = os.environ["MONGO_COL"].strip()
else:
    collection = "ragtagchunks"
client = pymongo.MongoClient(conn)
db = client[database]
col = db[collection]

# Load users from .env file
if "USERS" in os.environ:
    users_string = os.environ["USERS"]
    users = json.loads(users_string)
else:
    users = None

# Make it pretty because I can't :(
Bootstrap(app)

# Flask forms is magic
class ChunkForm(FlaskForm):
    chunk_question = StringField('Question', validators=[DataRequired()])
    chunk_answer = TextAreaField('Answer', validators=[DataRequired()])
    chunk_enabled = SelectField('Enabled', choices=[(True, True), (False, False)])
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
    llm_system = TextAreaField('System Message', validators=[DataRequired()])
    llm_prompt = TextAreaField('Prompt', validators=[DataRequired()])
    llm_temp = FloatField("Temperature", validators=[DataRequired()])
    llm_tokens = IntegerField("Number of tokens from LLM", validators=[DataRequired()])
    submit = SubmitField('Submit')

# Return embedding with instruction and text
def get_embedding(ins, text):
    return instructor_model.encode([[ins,text]]).tolist()[0]

# Return the retrieval augmented generative result
def get_rag(question, search_k, search_score_cut, llm_prompt, llm_system, llm_temp, llm_tokens):
    # Get all the chunks
    chunks = list(vector_search_chunks(question, search_k, search_score_cut))
    answer_scores = []

    # Build the LLM answer chunks and build up our answer scores for later
    answers = ""
    for answer in chunks:
        answers = answers + answer["chunk_answer"] + " "
        score_data = {"chunk_answer": answer["chunk_answer"], "score": answer["score"]}
        answer_scores.append(score_data)

    # Oh no! We have no chunks.  Just return a generic "we can't help you"
    # Score cut offs really help prevent LLM abuse.  This is your first guardrail.
    if answers == "":
        return {"input": "no chunks found", "output": "No data was found to answer this question", "chunks": {}}

    # Replace the template tokens with the question and the answers
    prompt = llm_prompt.replace("%q%", question)
    prompt = prompt.replace("%d%", answers)

    # One more replacement step to help our chat model out with a system prompt and proper control tokens
    llm_result = {}
    llm_result["input"] = prompt_format.replace("{prompt}", prompt)
    llm_result["input"] = llm_result["input"].replace("{system}", llm_system)

    # Generate LLM response and return the text
    llm_result["output"] = llama_model(llm_result["input"], max_tokens=llm_tokens, temperature=llm_temp)["choices"][0]["text"]

    # Find the baned tokens
    index = llm_result["output"].find(ban_token)

    # Check if the ban token is found in the string
    if index != -1:
        # Trim the string, including the marker and everything after it
        llm_result["output"] = llm_result["output"][:index]

    # Sure, throw the chunks in there too!
    llm_result["chunks"] = answer_scores

    return llm_result

# Count the number of tokens in the string (useful for setting limits)
def token_count(text):
    words = text.split()  # Split the text into words using spaces as the default delimiter
    return len(words)

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
                    "filter": { "equals": { "path": "chunk_enabled", "value": True}},
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
        if users != None:
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
    form = VectorSearchForm(search_k=DEFAULT_K, search_score_cut=DEFAULT_SCORE_CUT)
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
    # We're doing a vector search here
    chunks = []
    form = LLMForm(search_k=DEFAULT_K, search_score_cut=DEFAULT_SCORE_CUT, llm_prompt=str(DEFAULT_PROMPT), llm_system=str(DEFAULT_SYSTEM),llm_temp = DEFAULT_TEMP, llm_tokens=DEFAULT_TOKENS)
    if request.method == "POST":
        form_result = request.form.to_dict(flat=True)
        llm_response = get_rag(form_result["question"], form_result["search_k"], form_result["search_score_cut"], form_result["llm_prompt"], form_result["llm_system"], float(form_result["llm_temp"]), int(form_result["llm_tokens"]))
        return render_template('llm.html', chunks=llm_response["chunks"], form=form, llm_response=llm_response["output"],prompt=llm_response["input"])

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
        # Change the text True/False to a proper python type - Thanks WTForms :(
        if form_result["chunk_enabled"] == "True":
            form_result["chunk_enabled"] = True
        if form_result["chunk_enabled"] == "False":
            form_result["chunk_enabled"] = False
        embed_text = form_result["chunk_question"] + " " + form_result["chunk_answer"]
        # Stuff the token count into the form result
        form_result["chunk_tokens"] = token_count(form_result["chunk_question"] + " " + form_result["chunk_answer"])
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
            form.chunk_enabled.data = bool(chunk["chunk_enabled"])
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

# This chunk is bad, we need to make it feel bad
@app.route('/chunk_delete/<id>')
@login_required
def chunk_delete(id):
    chunk_data = col.delete_one({'_id': ObjectId(id)})
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
    if not q:
        return {'error': 'No q parameter found. You must ask a question - /api/rag/q=<string>'}
    if ((key != api_key) and (api_key != None)):
        return {'error': 'API key does not match'}

    # Get the LLM result for the question with default settings then return it
    llm_response = get_rag(q, DEFAULT_K, DEFAULT_SCORE_CUT, DEFAULT_PROMPT, DEFAULT_SYSTEM, DEFAULT_TEMP, DEFAULT_TOKENS)
    return llm_response 

# API endpoint for that mimics the response of the previous
# To respond quickly for api testing
@app.route('/api/ragfake')
def api_ragfake():
    key = request.args.get("key")
    q = request.args.get("q")
    
    # Make sure we have a valid key and question
    if not q:
        return {'error': 'No q parameter found. You must ask a question - /api/rag/q=<string>'}
    if ((key != api_key) and (api_key != None)):
        return {'error': 'API key does not match'}

    resp = {}
    resp["input"] = q
    resp["output"] = "this is a mock response from the api."
    resp["chunks"] = []

    for i in range(3):
        c = {}
        c["chunk_answer"] = "This is chunk answer number " + str(i)
        c["score"] = random.random()
        resp["chunks"].append(c)
    
    return resp


# API endpoint for getting a text embedding from instructor.
@app.route('/api/vector')
def api_vector():
    key = request.args.get("key")
    q = request.args.get("q")
    
    # Make sure we have a valid key and question
    if not q:
        return {'error': 'No q parameter found. You must provide a string to vectorize - /api/vector/q=<string>'}
    if ((key != api_key) and (api_key != None)):
        return {'error': 'API key does not match'}
    
    # Get the vector result for the string
    return get_embedding("Represent the question for retrieving supporting documents:", q)

# API Endpoint to dump all stored chunks
@app.route('/api/list')
def api_list():
    key = request.args.get("key")
    # Make sure we have a valid key and question
    if ((key != api_key) and (api_key != None)):
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
    if not q:
        return {'error': 'No q parameter found. You must ask a question - /api/search/q=<string>'}
    if ((key != api_key) and (api_key != None)):
        return {'error': 'API key does not match'}
    
    chunks = search_chunks(q)
    return json.loads(json_util.dumps(chunks))

# API Endpoint to perform vector search
@app.route('/api/vector_search')
def api_vector_search():
    key = request.args.get("key")
    q = request.args.get("q")
    
    # Make sure we have a valid key and question
    if not q:
        return {'error': 'No q parameter found. You must ask a question - /api/search/q=<string>'}
    if ((key != api_key) and (api_key != None)):
        return {'error': 'API key does not match'}
    
    chunks = vector_search_chunks(q, DEFAULT_K, DEFAULT_SCORE_CUT)
    return json.loads(json_util.dumps(chunks))
