# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define environment variable
ENV NAME TetrisRL

# Run train.py when the container launches
CMD ["python", "train.py"]