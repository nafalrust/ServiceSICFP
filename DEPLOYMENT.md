# Deployment Guide for Hugging Face Spaces

## Prerequisites

1. Hugging Face account (https://huggingface.co)
2. Trained CatBoost model file (`catboost_bp_model.pkl`)

## Deployment Steps

### 1. Create a New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose:
   - **Name**: `blood-pressure-classifier` (or your preferred name)
   - **License**: Apache 2.0 or MIT
   - **SDK**: Docker
   - **Visibility**: Public or Private

### 2. Clone the Space Repository

```bash
# Install Git LFS first (if not already installed)
git lfs install

# Clone your space
git clone https://huggingface.co/spaces/YOUR-USERNAME/blood-pressure-classifier
cd blood-pressure-classifier
```

### 3. Copy Service Files

Copy all files from the `ServiceAI` directory:

```bash
# From your SICFinal directory
cp ServiceAI/app.py blood-pressure-classifier/
cp ServiceAI/feature_extractor.py blood-pressure-classifier/
cp ServiceAI/heart_rate_detector.py blood-pressure-classifier/
cp ServiceAI/requirements.txt blood-pressure-classifier/
cp ServiceAI/Dockerfile blood-pressure-classifier/
cp ServiceAI/README.md blood-pressure-classifier/
```

### 4. Add Your Trained Model

```bash
# Create models directory
mkdir -p blood-pressure-classifier/models

# Copy your trained model
cp ServiceAI/models/catboost_bp_model.pkl blood-pressure-classifier/models/
```

**Important**: If your model file is large (>10 MB), use Git LFS:

```bash
cd blood-pressure-classifier
git lfs track "*.pkl"
git add .gitattributes
```

### 5. Update Dockerfile (if needed)

Uncomment the model copy line in `Dockerfile`:

```dockerfile
# Copy model file
COPY models/catboost_bp_model.pkl models/catboost_bp_model.pkl
```

### 6. Commit and Push

```bash
cd blood-pressure-classifier
git add .
git commit -m "Initial deployment: Blood Pressure Classification Service"
git push
```

### 7. Wait for Build

- Hugging Face will automatically build and deploy your Docker container
- Build time: 3-10 minutes
- Check the "Logs" tab for build progress

### 8. Test Your Deployment

Once deployed, your service will be available at:
```
https://YOUR-USERNAME-blood-pressure-classifier.hf.space
```

Test the endpoints:
```bash
# Health check
curl https://YOUR-USERNAME-blood-pressure-classifier.hf.space/health

# Get info
curl https://YOUR-USERNAME-blood-pressure-classifier.hf.space/info
```

## Configuration Options

### Environment Variables

You can set environment variables in Hugging Face Spaces settings:

1. Go to your Space settings
2. Add variables:
   - `PORT`: 7860 (default, usually not needed)
   - `MODEL_PATH`: /app/models/catboost_bp_model.pkl

### Hardware

For better performance, you can upgrade to:
- **CPU Basic**: Free (default)
- **CPU Upgrade**: $0.03/hour
- **GPU**: For faster inference (usually not needed for this model)

## Gradio Interface (Optional)

If you want a web UI instead of API-only:

1. Install Gradio:
```bash
pip install gradio
```

2. Create `gradio_app.py`:
```python
import gradio as gr
import requests

def predict_bp(ppg_signal, age, height, weight, gender):
    response = requests.post(
        "http://localhost:7860/predict",
        json={
            "ppg": ppg_signal,
            "age": age,
            "height": height,
            "weight": weight,
            "gender": gender
        }
    )
    return response.json()

interface = gr.Interface(
    fn=predict_bp,
    inputs=[
        gr.Textbox(label="PPG Signal (comma-separated)"),
        gr.Number(label="Age"),
        gr.Number(label="Height (cm)"),
        gr.Number(label="Weight (kg)"),
        gr.Radio([0, 1], label="Gender (0=Male, 1=Female)")
    ],
    outputs="json",
    title="Blood Pressure Classifier"
)

interface.launch()
```

## Monitoring

### View Logs

In your Space page:
1. Click "Logs" tab
2. View real-time logs
3. Check for errors or performance issues

### Usage Analytics

Hugging Face provides:
- Request count
- Response times
- Error rates

## Updating Your Deployment

To update your service:

```bash
cd blood-pressure-classifier

# Make changes to your files
# ...

# Commit and push
git add .
git commit -m "Update: description of changes"
git push
```

The Space will automatically rebuild and redeploy.

## Troubleshooting

### Build Fails

1. Check Dockerfile syntax
2. Verify all files are copied
3. Check requirements.txt versions
4. View build logs for specific errors

### Model Not Loading

1. Verify model file exists in `models/`
2. Check file size (if >500MB, may have issues)
3. Ensure Git LFS tracked the file
4. Check logs for permission errors

### Service Not Responding

1. Check if PORT is set correctly (7860)
2. Verify health endpoint works
3. Check logs for Python errors
4. Ensure all dependencies installed

### Out of Memory

If the service crashes with OOM:
1. Optimize model size
2. Upgrade to CPU Upgrade tier
3. Add memory limits in Dockerfile

## Security Best Practices

1. **Don't commit sensitive data**: Use Hugging Face Secrets for API keys
2. **Validate inputs**: Already implemented in the service
3. **Rate limiting**: Consider adding if public
4. **CORS**: Already configured for wide access

## Cost Estimation

- **Free tier**: Perfect for testing and low traffic
- **Paid tier**: ~$20-50/month for moderate traffic
- **GPU**: Only if you need fast batch processing

## Support

- Hugging Face Docs: https://huggingface.co/docs/hub/spaces
- Community Forum: https://discuss.huggingface.co
- Discord: https://discord.gg/hugging-face

## Example Space

See example deployed spaces:
- https://huggingface.co/spaces/examples/docker-space

## Next Steps

After deployment:
1. Test all endpoints
2. Monitor logs for errors
3. Add API documentation
4. Share your Space URL
5. Consider adding a Gradio UI for demo purposes
