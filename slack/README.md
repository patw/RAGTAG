# RAGTAG Slackbot

This bot can join your existing slack servers and utilizes the /api/rag endpoint to produce answers for questions that people
can send by @ mentioning the bot.

## Installation

Install the dependencies

```pip install -r requirements.txt```

## Slack Configuration

* Go to https://api.slack.com/apps create a new application
* Configure the application in the basic information setting, give it a name and a slack icon along with a description
* Select which slack spaces you want to install the bot to
* Turn on socket mode from the Socket mode menu
* Under Install App get your Bot token (xoxb-*) you'll need this for your .env file!
* Under app home give your bot a display name and show bot as always online
* In Oauth and Permissions give the bot app_mentions:read and chat:write scopes
* In Event Subscriptions click on subscribe to bot events and add app_mention

Verify everything looks good in the app manifest YAML, here is an example:

```
display_information:
  name: RAGTAG
  description: A RAGTAG Bot
  background_color: "#416949"
  long_description: This bot is powered by the RAGTAG system providing responses from manually entered and curated chunks and using the mistral-7b model to respond.  You can find ragtag at https://ragtag.dungeons.ca/
features:
  bot_user:
    display_name: RAGTAG
    always_online: true
oauth_config:
  scopes:
    bot:
      - app_mentions:read
      - chat:write
settings:
  event_subscriptions:
    bot_events:
      - app_mention
  interactivity:
    is_enabled: true
  org_deploy_enabled: false
  socket_mode_enabled: true
  token_rotation_enabled: false
```

## Running Bot

Copy sample.env to .env and modify it to have the proper endpoint and tokens.  RAGTAG must already be running and configured
with working chunks and LLM output for this bot to do anything.

```python slack_bot.py```