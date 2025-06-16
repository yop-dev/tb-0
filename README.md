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

## Deployment to Vercel

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Login to Vercel:
```bash
vercel login
```

3. Deploy:
```bash
vercel
```

4. For production deployment:
```bash
vercel --prod
```

## Environment Variables

No environment variables are required for basic functionality.

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
- `recordings/`: Directory for storing recorded coughs
- `vercel.json`: Vercel configuration
- `requirements.txt`: Python dependencies

## Notes

- The app requires at least 5 cough recordings for analysis
- Audio processing is done using librosa (with fallback to soundfile)
- The model is based on MobileNetV4 with Res2TSM blocks 