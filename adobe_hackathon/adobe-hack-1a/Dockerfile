# Use slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set input/output folders (already volume-mounted)
VOLUME ["/app/input", "/app/output"]

# Run the Python script
CMD ["python", "app.py"]
