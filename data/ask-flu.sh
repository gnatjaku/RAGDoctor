curl -X POST http://localhost:8000/ask   -H "Content-Type: application/json"   -d '{
            "question": "Podaj wirusy wywołującą grypę sezonową?",
            "top_k": 3
  }'