#!/bin/bash

curl -w "@curl-format.txt" -o /dev/null http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "messages": [
            {"role": "user", "content": "Tell me about CIS 565 at UPenn."}
        ],
        "max_tokens": 1
    }'
