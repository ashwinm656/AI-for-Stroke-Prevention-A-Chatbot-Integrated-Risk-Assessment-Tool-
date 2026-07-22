# NeuroCare — integrated frontend (with real accounts)

Dark NeuroCare dashboard UI matching your inspo screenshot, wired to your
real model (age, gender, 15 symptoms → RandomForest), with actual
sign up / log in, saved assessment history per account, and a working
chatbot — not just a UI mockup.

## What's actually functional now
- **Sign up / log in** — real accounts (username, email, password), stored
  in a local SQLite database (`db.py` / `neurocare.db`), passwords hashed
  with salted PBKDF2 (stdlib `hashlib`, no extra dependency).
- **New analysis** — the Risk check form saves every submission to the
  logged-in user's history in the database (age, gender, symptoms, model
  result). It's not just displayed and thrown away.
- **Dashboard** — pulls the user's *real* latest result and a "your past
  checks" list from the database. Before you've run a first check, it shows
  the same sample numbers as your inspo screenshot; after that, it's your
  actual data.
- **Chatbot** — same HuggingChat integration as your original `app.py`,
  now reading credentials from `st.secrets` instead of being hardcoded.
  Works as long as `HF_EMAIL`/`HF_PASS` are set (see Secrets setup below).
- **News** — same as before, needs `NEWS_API_KEY` in secrets.

## ⚠️ One real limitation: where the database lives
`neurocare.db` is a plain file next to `app.py`. That's genuinely
persistent as long as the app keeps running, but:
- **Locally**: persists across runs, fine.
- **Streamlit Community Cloud**: the filesystem resets when the app
  redeploys or wakes up from sleeping — so accounts/history created there
  can get wiped. For a portfolio demo this is usually acceptable (just
  re-sign-up after a redeploy). For real persistence, swap `db.py`'s
  sqlite3 calls for a hosted database (Supabase/Postgres/etc.) — the
  function signatures (`create_user`, `verify_user`, `save_assessment`,
  `get_latest_assessment`, `get_history`, `get_stats`) are written so you
  can swap the internals without touching `app.py`.

## Files
- `app.py` — the app (auth gate → Dashboard / Risk check / News / Chatbot)
- `db.py` — SQLite auth + assessment-history layer
- `stroke_data.csv` — **synthetic** dataset I generated (35,000 rows) since
  you didn't have the original. Same schema `stroke_model.py` expects.
- `stroke_model.py` — your training script, lightly modified (see below)
- `stroke_model.pkl`, `scaler.pkl` — trained on the synthetic data above
- `requirements.txt`
- `secrets.toml.example` — template for chatbot/news API keys

## ⚠️ The model is trained on synthetic data
I generated a synthetic dataset with plausible relationships (older age +
more/worse symptoms → higher risk) since I don't have your real patient
data, and trained your unmodified model architecture on it. **This is a
working placeholder, not a validated clinical tool.** Swap in your real
`stroke_data.csv` and rerun `python stroke_model.py` when you have it —
that regenerates `stroke_model.pkl`/`scaler.pkl`.

I also capped the RandomForest (`max_depth=10`, `n_estimators=150`,
`min_samples_leaf=5`) — the default unbounded settings produced an 86MB
pickle, too big to push comfortably to GitHub. This version is ~10MB.

## ⚠️ Rotate your HuggingFace password
Your original `app.py` had your HuggingChat email + password hardcoded in
plain text. If this repo is public, please change that password now. The
new app reads credentials from `st.secrets` instead.

## Secrets setup (chatbot + news)
Copy `secrets.toml.example` to `.streamlit/secrets.toml` and fill in:
```
NEWS_API_KEY = "..."
HF_EMAIL = "..."
HF_PASS = "..."
```
Never commit the filled-in `secrets.toml`. On Streamlit Community Cloud,
paste the same values into your app's Settings → Secrets instead.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
First run creates `neurocare.db` automatically — sign up for an account
in the app, then use Risk check to log a real assessment.

## Deploy to Streamlit Community Cloud
1. Push everything in this folder to your GitHub repo (add `neurocare.db`
   to `.gitignore` — no need to commit a local user database).
2. Go to https://share.streamlit.io → "New app" → point to `app.py`.
3. Add your secrets under the app's Settings → Secrets.
4. Deploy.
