#!/bin/bash

pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --break-system-packages
flask run -p 80 --host 0.0.0.0