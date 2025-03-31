# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install additional dependencies for GPU support
RUN pip install torch transformers

# Make port 80 available to the world outside this container
EXPOSE 80 22

# Run the application
CMD ["python", "infer.py"]