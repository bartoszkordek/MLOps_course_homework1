# MLOps Course – Homework 1  🚀

## 📘 Project Description

This project was developed as part of an **MLOps** course and demonstrates a complete Machine Learning workflow — from 
data processing to model training and evaluation.

MLOps combines **Machine Learning, DevOps, and Data Engineering** practices to build scalable, reproducible, and 
production-ready ML systems.

---
## 📌 Project Scope

The goal of this project is to build a **production-ready Machine Learning inference service** based on a pre-trained 
model.  
The project focuses on:
- integrating a pre-trained Machine Learning model into an application  
- exposing model predictions via a REST API  
- implementing automated tests for the application  
- containerizing the service using Docker and orchestrating it with Docker Compose  
- structuring the project according to MLOps best practices  

This project does not include model training or experimentation, as the model was provided.
Instead, the focus is on the operational side of Machine Learning (MLOps).

---
## 🏗️ Project Structure
.
├── data/               # Raw and processed data  
├── notebooks/          # Jupyter notebooks (experiments)  
├── src/                # Source code  
│   ├── data/           # Data processing  
│   ├── features/       # Feature engineering  
│   ├── models/         # Training and inference  
│   └── utils/          # Helper functions  
├── tests/              # Unit and API tests  
├── requirements.txt    # Dependencies  
└── README.md  
---

## ⚙️ Requirements
- uv
- Docker
- Docker Compose
### `uv` installation:

```bash
pip install uv
```
or (macOS / Linux):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or via Homebrew
```bash
brew install uv
```

---
## 🔧 Installation
Clone repository
```bash
git clone https://github.com/bartoszkordek/MLOps_course_homework1.git
cd MLOps_course_homework1
```
Verify `uv` installation by running:
```bash
uv --version
```

Initialize the Python project in a current directory
```bash
uv venv --python 3.12
uv init
```

Activate environment:
```bash
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate   # Windows
```
To deactivate environment:
```bash
deactivate
```

Download and unpack the model shared with Google Drive: https://drive.google.com/file/d/1NRZdYq5jweVRUzAZG518LMhs4E56IgxG/view?usp=share_link
in /models/ folder.

Run the actual application server, exposed on port 8000:
```bash
uv run uvicorn app:app --reload --port 8000
```

---
## 🔮 Using the API

### Server address

```bash
curl http://localhost:8000/
```
**Response:**
```json
{
  "message": "Welcome to the ML API" 
}
```
### System health check
```bash
curl -X GET "http://localhost:8000/health"
```
**Sample response:**
```json
{
  "status":"ok"
}
```

### Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What a great MLOps lecture, I am very satisfied"
  }'
```

Sample response:
```json
{
  "prediction": "positive"
}
```

---
## 🧪 Unit Testing
The project includes an extensive set of **unit tests** using [pytest](https://docs.pytest.org/).  

Run the tests using the following command:
```bash
uv run pytest tests -rP
```

## 📊 Model

The project uses a pre-trained Machine Learning model provided as part of the assignment.

- the model is loaded and used for inference only  
- no additional training or tuning is performed  
- the application is responsible for serving predictions via API  

The main responsibility of this project is to ensure that the model:

- can be reliably loaded  
- produces predictions correctly  
- is accessible through a well-defined interface  

---
## 🐳 Containerization

The application is containerized using Docker, which ensures:

- consistent runtime environment
- easy deployment
- reproducibility across machines

Build the Docker image:
```bash
docker build -t sentiment-analysis-app .
```

Run the Docker container:
```bash
docker run --rm -p 8000:8000 sentiment-analysis-app
```

To stop the container, press Ctrl+C in the terminal where it's running or:
```bash
docker ps -a # then <copy container-id> you want to delete
docker stop <container-id>
```

or using Docker Compose:
```bash
docker compose up
```

Turn off with
```bash
docker compose down
```
---
## 🧠 Tech Stack
- **Python 3.12** – core language

### API & Serving
- **FastAPI** – REST API for model inference
- **Uvicorn** – ASGI server
- 
### Machine Learning
- **scikit-learn** – model inference
- **PyTorch** – backend for embeddings
- **sentence-transformers** – text embeddings
- **joblib** – model serialization

### Data Processing
- **clean-text** – text preprocessing

### Testing
- **pytest** – test framework
- **httpx** – API testing

### Code Quality
- **ruff** – linting & formatting
- **mypy** – static type checking
- **pre-commit** – git hooks

### Environment & Packaging
- **pyproject.toml** – dependency and project management
- **uv** – package installer (with custom PyTorch index)

### Containerization
- **Docker** – containerized deployment
- **Docker Compose** - orchestration
---
## 📖 API Documentation
Once the server is running, the interactive documentation is available at:
**Swagger UI**: http://127.0.0.1:8000/docs

---
## 🪪 License
This project is released under the MIT License.
