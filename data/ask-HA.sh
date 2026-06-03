curl -X POST http://localhost:8000/ask   -H "Content-Type: application/json"   -d '{
            "question": "Jak przebiega typowy zawał?",
            "top_k": 3
  }'