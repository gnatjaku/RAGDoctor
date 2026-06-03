curl -X POST http://localhost:8000/ask   -H "Content-Type: application/json"   -d '{
            "question": "W jaki sposób można zarazić się wirusem HIV?",
            "top_k": 1
  }'