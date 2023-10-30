# Basic flask stuff for building http APIs and rendering html templates
from flask import Flask, render_template, redirect, url_for, request, session

# Bootstrap integration with flask so we can make pretty pages
from flask_bootstrap import Bootstrap

# Flask forms integrations which save insane amounts of time
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, SelectField, PasswordField, IntegerField, FloatField
from wtforms.validators import DataRequired

# Instructor-large embedding model for creating vectors
from InstructorEmbedding import INSTRUCTOR
instructor_model = INSTRUCTOR('hkunlp/instructor-large')

# Use the wonderful llama.cpp library to execute our LLM (mistral-7b with dolphin fine tune)
from llama_cpp import Llama
llama_model = Llama(model_path="dolphin-2.1-mistral-7b.Q5_K_S.gguf")
system_message = "You are a helpful assistant who will always answer the question with only the data provided and in 3 sentences."
prompt_format = "<|im_start|>system\n" + system_message + "<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant:"

print("Preloaded big files")