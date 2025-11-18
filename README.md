---
title: KhabarGPT API
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: fastapi
app_file: main.py
---

# KhabarGPT: Ensemble Fake News Detector API

This Space hosts the FastAPI backend for the Semester 7 Capstone Project. It provides the `/analyze` endpoint, which uses a 3-method ensemble voting system:

1.  **Custom TF-IDF Model:** For Indian-context prediction (BharatKosh).
2.  **Web Search:** For live fact-checking verification.
3.  **General BERT Model:** For linguistic style analysis.

To access the API documentation (Swagger UI), append `/docs` to the Space URL.
