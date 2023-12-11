#!/bin/bash

echo "------------------"
echo "     Network      "
echo "------------------"
for i in {1..10}
do
    # dummy bad requests
    curl -w "@curl-format.txt" -o /dev/null http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "facebook/opt-6.7b",
            "messages": [
                {"role": "user", "content": "Tell me about CIS 565 at UPenn."}
            ],
            "max_tokens": 1,
        }'
done

echo "------------------"
echo "     No cache     "
echo "------------------"
curl -w "@curl-format.txt" -o /dev/null http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-6.7b",
        "messages": [
            {"role": "user", "content": "Tell me about CIS 565 at UPenn."}
        ],
        "max_tokens": 1
    }'

echo "------------------"
echo "     With cache   "
echo "------------------"
curl -w "@curl-format.txt" -o /dev/null http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-6.7b",
        "messages": [
            {"role": "user", "content": "Tell me about CIS 565 at UPenn."}
        ],
        "max_tokens": 1
    }'
