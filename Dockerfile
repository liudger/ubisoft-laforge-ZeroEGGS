FROM continuumio/miniconda3

WORKDIR /app

# Create a conda environment and activate it
RUN conda create -n zeggs python=3.8
SHELL ["conda", "run", "-n", "zeggs", "/bin/bash", "-c"]

# Install PyTorch and other dependencies
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install additional dependencies
RUN apt-get update && apt-get -y install sox ffmpeg

# Install other dependencies from requirements.txt
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set Python to run in unbuffered mode to see logs immediately
ENV PYTHONUNBUFFERED=1

# Set logging level to DEBUG
ENV LOG_LEVEL=DEBUG

# Expose the websocket port
EXPOSE 8000

# Command to run the app with debug logging
CMD ["conda", "run", "-n", "zeggs", "python", "-u", "app.py"]