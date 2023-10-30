#!/bin/bash

source .env
echo "Using conn string ${MONGO_CON}"
echo "Using key ${SECRET_KEY}"

echo
echo "+================================"
echo "| START: RAGTAG Service"
echo "+================================"
echo

datehash=`date | md5sum | cut -d" " -f1`
abbrvhash=${datehash: -8}

if [ ! -f dolphin-2.1-mistral-7b.Q5_K_S.gguf ]
    then
    echo 
    echo "Downloading prereqs"
    echo
    wget https://huggingface.co/TheBloke/dolphin-2.1-mistral-7B-GGUF/resolve/main/dolphin-2.1-mistral-7b.Q5_K_S.gguf
fi

echo 
echo "Building container using tag ${abbrvhash}"
echo
docker build -t graboskyc/ragtag:latest -t graboskyc/ragtag:${abbrvhash} .

EXITCODE=$?

if [ $EXITCODE -eq 0 ]
    then

    echo 
    echo "Starting container"
    echo
    docker stop ragtag
    docker rm ragtag
    docker run -t -i -d -p 80:80 --name ragtag -e "SPECUIMDBCONNSTR=${MONGO_CON}" graboskyc/ragtag:latest

    echo
    echo "+================================"
    echo "| END:  RAGTAG Service"
    echo "+================================"
    echo

else
    echo
    echo "+================================"
    echo "| ERROR: Build failed"
    echo "+================================"
    echo
fi
