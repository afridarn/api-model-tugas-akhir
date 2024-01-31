# Klasifikasi Larva Nyamuk - API Server

## Description

This FastAPI application serves as an API for a machine learning model designed to predict mosquito larvae microscopic images.

## Environment and Library

- Python v3.11.6
- pip v23.2.1
- FastAPI v0.105.0
- TensorFlow v2.15.0
- Uvicorn v0.25.0

## Prerequisites

- Python 3.11.6

If Python v3.11 is not yet installed on the system, follow these steps:

1. Visit the official python website to download the python installer https://www.python.org/downloads/
2. Choose the 3.11 version and follow the provided installation instructions.

## Installation

Step-by-step instructions to install this app:

1. **Clone the Repository:**

   ```
   git clone https://github.com/afridarn/api-model-tugas-akhir.git
   cd api-model-tugas-akhir
   ```

2. **Download the Model File**

- Create folder named **model** in the project directory
- Download the model file from the https://its.id/m/model
- Place the downloaded model into a folder named model

3. **Create a Virtual Environment**

   ```
   python -m venv venv
   cd ./Scripts
   . activate
   cd ..
   ```

4. **Install Dependencies**
   ```
   pip install fastapi tensorflow==2.15 uvicorn
   ```

## Running the App

```
uvicorn main:app
```

The application will be accessible at http://localhost:8000
