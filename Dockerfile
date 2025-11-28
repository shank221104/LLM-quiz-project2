FROM python:3.11-slim

WORKDIR /code

# System deps for Playwright + matplotlib
RUN apt-get update && apt-get install -y \
    wget gnupg libnss3 libatk-bridge2.0-0 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libasound2 libatspi2.0-0 libxshmfence-dev libcairo2 libpango-1.0-0 \
    libgdk-pixbuf-xlib-2.0-0 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Chromium for Playwright (no --with-deps, so no ttf-* fonts)
RUN python -m playwright install chromium

# Copy app source
COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
