FROM continuumio/miniconda3
# docker pull continuumio/miniconda3:23.5.2-0-alpine
# docker pull continuumio/miniconda3:latest
# docker pull continuumio/miniconda3:master-alpine

WORKDIR /app

# Create a conda environment and activate it
RUN conda create -n zeggs python=3.8
SHELL ["conda", "run", "-n", "zeggs", "/bin/bash", "-c"]

# Install PyTorch and other dependencies
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install additional dependencies
RUN apt-get update && apt-get -y install sox ffmpeg

# Install other dependencies from requirements.txt
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the Flask app port
EXPOSE 5000

# Command to run the Flask app
CMD ["conda", "run", "-n", "zeggs", "python", "app.py", "--host", "0.0.0.0"]