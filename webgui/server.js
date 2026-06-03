const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });
const express = require('express');
const axios = require('axios');

const app = express();
const PORT = process.env.WEBGUI_PORT || 3000;
const RAG_API_URL = process.env.RAG_API_URL || 'http://localhost:8000';

app.use(express.json({ limit: '2mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Proxy endpoint: /api/ask → RAGDoctor FastAPI
app.post('/api/ask', async (req, res) => {
    const { question, top_k } = req.body;
    if (!question || !question.trim()) {
        return res.status(400).json({ error: 'Pytanie nie może być puste.' });
    }
    try {
        const response = await axios.post(`${RAG_API_URL}/ask`, {
            question: question.trim(),
            top_k: top_k || null,
        }, { timeout: 300000 });
        res.json(response.data);
    } catch (err) {
        if (err.response) {
            res.status(err.response.status).json({ error: err.response.data?.detail || 'Błąd API' });
        } else {
            res.status(503).json({ error: `Nie można połączyć z RAGDoctor API: ${err.message}` });
        }
    }
});

// Proxy endpoint: /api/answers/pdf -> RAGDoctor FastAPI
app.post('/api/answers/pdf', async (req, res) => {
    const { answer, question, citations, fileName } = req.body || {};
    if (!answer || !String(answer).trim()) {
        return res.status(400).json({ error: 'Treść odpowiedzi nie może być pusta.' });
    }

    try {
        const response = await axios.post(
            `${RAG_API_URL}/answers/pdf`,
            {
                answer: String(answer).trim(),
                question: question || null,
                citations: Array.isArray(citations) ? citations : [],
                file_name: fileName || null,
            },
            {
                responseType: 'arraybuffer',
                timeout: 120000,
            }
        );

        res.setHeader('Content-Type', response.headers['content-type'] || 'application/pdf');
        res.setHeader(
            'Content-Disposition',
            response.headers['content-disposition'] || 'attachment; filename="ragdoctor-odpowiedz.pdf"'
        );
        res.send(Buffer.from(response.data));
    } catch (err) {
        if (err.response) {
            let message = 'Błąd API';
            try {
                const body = JSON.parse(Buffer.from(err.response.data).toString('utf-8'));
                message = body.detail || body.error || message;
            } catch {
                message = err.response.statusText || message;
            }
            res.status(err.response.status).json({ error: message });
        } else {
            res.status(503).json({ error: `Nie można połączyć z RAGDoctor API: ${err.message}` });
        }
    }
});

// Health check
app.get('/api/health', async (req, res) => {
    try {
        const response = await axios.get(`${RAG_API_URL}/health`, { timeout: 5000 });
        res.json({ webgui: 'ok', ragapi: response.data });
    } catch {
        res.status(503).json({ webgui: 'ok', ragapi: 'unavailable' });
    }
});

app.listen(PORT, () => {
    console.log(`🩺 RAGDoctor WebGUI running at http://localhost:${PORT}`);
    console.log(`   RAG API → ${RAG_API_URL}`);
});

