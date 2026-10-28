FROM python:3.11-slim

# Create user with UID 1000 (standard for Hugging Face Spaces and secure production cloud hosting)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install dependencies as user
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user . .

# Expose standard cloud port
EXPOSE 7860

# Launch ConnectTel Production API & Web App
CMD ["python", "app.py"]
