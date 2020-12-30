#!/bin/bash

echo "Starting server... This could take a while, as we are loading machine learning models for the server."

source env/bin/activate && python src/index.py