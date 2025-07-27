# Use a specific Python version for consistency
FROM --platform=linux/amd64 python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# This command will be executed when the container runs
# Note: The user will provide the collection directory when they run the container
ENTRYPOINT ["python", "run_analysis.py"]
