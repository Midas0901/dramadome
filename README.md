# DramaDome — Hybrid Recommendation Web App

A Flask app that recommends Asian dramas by combining a user's structured profile (preferred genre / country / watch history) with a free-text mood query.

## Project structure (Flask requires this exact layout)
dramadome/
├── app.py              <- main Flask application
├── requirements.txt    <- Python dependencies
├── Procfile            <- tells the host how to start the app
├── templates/          <- ALL .html files must live here (Flask looks here by default)
│   ├── index.html
│   ├── for_you.html
│   ├── recommend.html
│   ├── results.html
│   ├── user.html
│   └── users.html
├── static/             <- CSS / JS / images (Flask looks here by default)
│   └── style.css
└── data/
├── dramas.csv
└── users.csv
text## Run locally

```bash
pip install -r requirements.txt
python app.py
Then open http://127.0.0.1:5000 in your browser.
Deploy so it stays online (Render.com, free tier)

Push this whole folder to a GitHub repo.
Go to https://render.com → New → Web Service → connect your GitHub repo.
Build command: pip install -r requirements.txt
Start command: gunicorn app:app
Deploy. Render gives you a permanent URL like https://dramadome.onrender.com.

(GitHub itself only hosts code — GitHub Pages can't run Python/Flask, only static HTML/CSS/JS. Render/Railway/PythonAnywhere are the free options that actually run Flask continuously.)
