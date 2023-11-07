# This library is the newest/best method of talking to Slack - THIS MAY CHANGE
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Requests and regular expressions
import requests
import re

# Nice way to load environment variables for deployments
from dotenv import load_dotenv
import os

# Get environment variables
load_dotenv()

bot_token = os.environ["BOT_TOKEN"]
app_token = os.environ["APP_TOKEN"]
ragtag_endpoint = os.environ["RAGTAG_ENDPOINT"]
ragtag_key = os.environ["RAGTAG_KEY"]

app = App(token=bot_token)

# Remove the @mention to the bot
def strip_user(text):
    pattern = r'<@(.*?)>'
    replacement = ''
    return re.sub(pattern, replacement, text)

@app.event("app_mention")
def handle_app_mention_events(body, say):
    req_user = body["event"]["user"]
    message = strip_user(body["event"]["text"])
    response = requests.get(ragtag_endpoint + "?key=" + ragtag_key + "&q=" + message)
    data = response.json()
    say("<@" + req_user + "> " + data["output"])

if __name__ == "__main__":
    SocketModeHandler(app, app_token=app_token).start()

