# Use an official Python runtime as a parent image
FROM python:3.7-slim

# Set the working directory to /app
WORKDIR /

# Copy the current directory contents into the container at /app
COPY / /

# Install any needed packages specified in requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg -y
RUN pip install --trusted-host pypi.python.org -r requirepip.txt

# Run app
ENTRYPOINT ["python", "app/src/mountain_car_qlearn.py"]
#CMD ["python", "app/src/mountain_car_qlearn.py"]