# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir torch gymnasium numpy

# Define environment variable
ENV NAME TetrisRL

# Run train.py when the container launches
CMD ["python", "train.py"]