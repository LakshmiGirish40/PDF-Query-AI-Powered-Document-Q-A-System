
# Use Python 3.12-slim as the base image
FROM python:3.12-slim

# Set a working directory for your application (using /app is common)
WORKDIR /app

COPY RAG_Document_Intelligent/requirements.txt ./requirements.txt


# Install Python dependencies
RUN pip install --no-cache-dir -r RAG_Document_Intelligent/requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Expose port 8007 so the container can communicate on this port
EXPOSE 9000

# Define the command to run your application
CMD ["python", "RAG_Document_Intelligent/run.py"]

