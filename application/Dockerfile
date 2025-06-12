FROM python:3.10
WORKDIR /src

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /src/

# Install dependencies (will only re-run if requirements.txt changes)
RUN pip install -r requirements.txt

# Now copy the rest of the code
COPY . /src

EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]