# Dockerfile
FROM python:3.8-slim

# Set a directory for the app
WORKDIR /usr/src/app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Command to run the application
CMD ["python", "./app.py"]