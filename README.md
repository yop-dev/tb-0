# TB Cough Detection Web App

A web application for detecting tuberculosis from cough sounds using deep learning.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask app:
```bash
python app.py
```

3. Open http://localhost:5000 in your browser

## Deployment to Render (Free-tier)

1. Go to [https://render.com](https://render.com) and log in or sign up.

2. Click **"New Web Service"** and connect your GitHub repository.

3. Select the repo for this project when prompted.

4. Fill in the deployment settings:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
   - **Instance Type:** Free Plan

5. Click **"Create Web Service"** and wait for the deployment to finish.

6. After successful deployment, Render will provide you with a public URL (e.g., `https://your-app.onrender.com`).


## Environment Variables

No environment variables are required for basic functionality.

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
- `recordings/`: Directory for storing recorded coughs
- `render.yaml`: Vercel configuration
- `requirements.txt`: Python dependencies

## Notes

- The app requires at least 5 cough recordings for analysis
- Audio preprocessing is handled using librosa, with a fallback to soundfile for compatibility.
- The underlying model is a MobileNetV4 architecture enhanced with Res2TSM blocks, designed for efficient temporal feature extraction.
- Depending on the Render hosting tier, the app will automatically load either a quantized (for lower-memory tiers) or full-precision model (for higher tiers).
