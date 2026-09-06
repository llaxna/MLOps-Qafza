# MLOps Qafza — Machine Learning Operations Journey

## About the Course

This repository contains my work and practical projects for the **Qafza MLOps Course**, a 12-week learning journey focused on taking machine learning models from development to production.

The course covers the complete MLOps lifecycle, starting with building reliable machine learning pipelines and progressing toward APIs, containers, data pipelines, versioning, experiment tracking, distributed training, feature stores, monitoring, automated retraining, and infrastructure as code.

The final goal is to combine the knowledge from all phases into an **end-to-end production-ready ML system**.

---
# MLOps Qafza — Machine Learning Operations Journey

## About the Course

This repository contains my practical work and projects for the **Qafza MLOps Course**, a 12-week learning journey focused on taking machine learning models from development to production.

The course covers the complete MLOps lifecycle, progressing from reliable machine learning pipelines to APIs, containerization, data pipelines, versioning, experiment tracking, distributed training, feature stores, monitoring, automated retraining, and infrastructure as code.

The final goal is to combine these concepts into an **end-to-end production-ready ML system**.

---

# Course Roadmap

| Week               | Phase                  | Topic                     | Core Tools                | Main Focus                                        |
| ------------------ | ---------------------- | ------------------------- | ------------------------- | ------------------------------------------------- |
| **1**              | Local Foundations      | Leakage-proof ML Pipeline | Python, Scikit-learn      | Build a reliable ML pipeline without data leakage |
| **2**              | Local Foundations      | Deep Learning Pipeline    | PyTorch, Hugging Face     | Train and prepare a DL model for production       |
| **3**              | Production APIs        | Production API            | FastAPI, Pydantic         | Serve a model through a validated REST API        |
| **4**              | Containerization       | Docker                    | Docker Engine             | Package the application into a portable container |
| **5**              | Data Pipelines         | ETL Pipeline              | Python ETL, Database      | Automate data ingestion and transformation        |
| **6**              | Data Versioning        | Versioning                | DVC                       | Track datasets and model artifacts                |
| **7**              | Experiment Tracking    | MLflow                    | MLflow Registry           | Track and compare ML experiments                  |
| **8**              | Distributed Training   | Distributed ML            | Ray Core, Ray Train       | Scale model training across resources             |
| **9**              | Feature Store          | Feature Management        | Feast, Ray                | Build reusable production features                |
| **10**             | Monitoring             | Monitoring                | Prometheus, Grafana       | Monitor model and system health                   |
| **11**             | Continuous Retraining  | Automation                | Ray Serve, GitHub Actions | Automate retraining and deployment                |
| **12**             | Infrastructure as Code | Infrastructure            | Terraform                 | Provision infrastructure programmatically         |
| **Final Capstone** | Capstone               | End-to-End ML System      | Complete Stack            | Build and present a production-ready MLOps system |

---

# Course Learning Path

The course progressively builds an ML system through the following stages:

```text
Machine Learning
      │
      ▼
Deep Learning
      │
      ▼
Model API
      │
      ▼
Docker
      │
      ▼
Data Pipeline
      │
      ▼
Data & Model Versioning
      │
      ▼
Experiment Tracking
      │
      ▼
Distributed Training
      │
      ▼
Feature Store
      │
      ▼
Monitoring
      │
      ▼
Automated Retraining
      │
      ▼
Infrastructure as Code
      │
      ▼
END-TO-END MLOps SYSTEM
```

The objective is to understand not only how to train a model, but how to **build, package, deploy, monitor, reproduce, and continuously improve** machine learning systems.

---

# Repository Structure

The repository is organized by course tasks and supporting project files.

```text
MLOps-Qafza/
│
├── Tasks/
│   │
│   ├── Task01.ipynb
│   │
│   ├── Task02/
│   │   ├── 01_read_and_join.ipynb
│   │   ├── 02_create_label.ipynb
│   │   ├── 03_train_validation_test_split.ipynb
│   │   ├── 04_eda.ipynb
│   │   ├── 05_feature_engineering.ipynb
│   │   ├── 06_train_models.ipynb
│   │   └── README.md
│   │
│   └── ...
│
├── artifacts/
│   └── generated datasets, models, charts and results
│
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Week 1 — Leakage-Proof ML Pipeline

### Topic

Leakage-proof Machine Learning Pipeline

### Core Tools

* Python
* Scikit-learn
* Pandas
* NumPy

### Objective

Build a clean machine learning pipeline that avoids data leakage and produces reproducible results.

The main focus is understanding the complete ML workflow:

```text
Raw Data
   ↓
Data Preparation
   ↓
Label Creation
   ↓
Train / Validation / Test Split
   ↓
EDA
   ↓
Feature Engineering
   ↓
Preprocessing
   ↓
Model Training
   ↓
Evaluation
   ↓
Model Persistence
```

### Suggested Task

> Create a complete Scikit-learn pipeline with preprocessing, cross-validation, and model persistence.

### Practical Project

The Week 1 work uses the **Olist Brazilian E-Commerce dataset** to build a model that predicts whether an order will be delivered late.

---

# Task 02 — Olist Late Delivery Prediction

Task 02 applies the Week 1 concepts to the **Olist Brazilian E-Commerce dataset**.

The objective is to predict whether an order will be:

```text
On Time
   vs.
Late
```

The task is divided into six notebooks:

| Notebook                               | Purpose                                                   |
| -------------------------------------- | --------------------------------------------------------- |
| `01_read_and_join.ipynb`               | Build the order-level dataset from the source tables      |
| `02_create_label.ipynb`                | Create the late-delivery target                           |
| `03_train_validation_test_split.ipynb` | Create chronological train, validation, and test sets     |
| `04_eda.ipynb`                         | Explore the training data and identify important patterns |
| `05_feature_engineering.ipynb`         | Create and preprocess model-ready features                |
| `06_train_models.ipynb`                | Train, tune, compare, and evaluate models                 |

The task demonstrates important ML concepts including:

* Relational data preparation
* Target creation
* Time-based data splitting
* Exploratory data analysis
* Feature engineering
* Data leakage prevention
* Handling imbalanced classification
* Model comparison and tuning
* Model persistence

The selected model is a **Logistic Regression classifier**, chosen based on its performance for the minority **Late** class.

---

# Week 2 — Deep Learning Pipeline

### Topic

Deep Learning Pipeline

### Core Tools

* PyTorch
* Hugging Face

### Objective

Train a deep learning model and prepare it for production.

### Suggested Task

> Train a PyTorch model and export it to ONNX.

Key concepts:

* neural networks
* training loops
* validation
* model serialization
* ONNX
* production inference

---

# Week 3 — Production API

### Topic

Production API

### Core Tools

* FastAPI
* Pydantic

### Objective

Expose the machine learning model through a production-style REST API.

### Suggested Task

> Build a FastAPI inference service with input validation.

Expected workflow:

```text
Client Request
      ↓
FastAPI
      ↓
Pydantic Validation
      ↓
Feature Preprocessing
      ↓
ML Model
      ↓
Prediction
      ↓
API Response
```

---

# Week 4 — Containerization

### Topic

Docker

### Core Tool

* Docker Engine

### Objective

Package the application and its dependencies into a portable container.

### Suggested Task

> Containerize the API and model using Docker.

Key concepts:

* Dockerfile
* images
* containers
* dependency management
* reproducible environments
* containerized inference

---

# Week 5 — Data Pipelines

### Topic

ETL Pipeline

### Core Tools

* Python ETL
* Database

### Objective

Automate the movement and transformation of data.

### Suggested Task

> Build an ETL pipeline connected to a database.

Expected workflow:

```text
Source Data
    ↓
Extract
    ↓
Transform
    ↓
Validate
    ↓
Load
    ↓
Database
```

---

# Week 6 — Data Versioning

### Topic

Data and Model Versioning

### Core Tool

* DVC

### Objective

Make datasets and model artifacts reproducible and traceable.

### Suggested Task

> Version datasets and model artifacts using DVC.

Key concepts:

* dataset versioning
* model versioning
* reproducibility
* experiment lineage
* artifact tracking

---

# Week 7 — Experiment Tracking

### Topic

MLflow

### Core Tool

* MLflow Registry

### Objective

Track experiments and compare different model runs.

### Suggested Task

> Log multiple experiments with MLflow.

Track information such as:

* parameters
* metrics
* models
* artifacts
* experiment versions

Expected workflow:

```text
Experiment
    ↓
Parameters
    ↓
Training
    ↓
Metrics
    ↓
MLflow
    ↓
Model Registry
```

---

# Week 8 — Distributed Training

### Topic

Distributed Machine Learning

### Core Tools

* Ray Core
* Ray Train

### Objective

Scale machine learning training across available computational resources.

### Suggested Task

> Train the model using Ray Train.

Key concepts:

* distributed execution
* parallel training
* resource management
* scalable experimentation

---

# Week 9 — Feature Store

### Topic

Feature Management

### Core Tools

* Feast
* Ray

### Objective

Create reusable and consistent production features.

### Suggested Task

> Build a Feature Store using Feast.

The Feature Store becomes the central location for managing features used by training and inference.

```text
Raw Data
    ↓
Feature Engineering
    ↓
Feature Store
    ├── Training
    └── Inference
```

---

# Week 10 — Monitoring

### Topic

Model and System Monitoring

### Core Tools

* Prometheus
* Grafana

### Objective

Monitor the health and behavior of the production ML system.

### Suggested Task

> Create monitoring dashboards and alerts.

Potential monitoring areas:

* API latency
* request volume
* errors
* CPU / memory
* prediction distribution
* model performance
* data drift
* feature drift

---

# Week 11 — Continuous Retraining

### Topic

Automation

### Core Tools

* Ray Serve
* GitHub Actions

### Objective

Automate model retraining and deployment.

### Suggested Task

> Implement webhook-triggered retraining.

Expected workflow:

```text
New Data
   ↓
Trigger
   ↓
ETL
   ↓
Feature Engineering
   ↓
Training
   ↓
Evaluation
   ↓
Model Registry
   ↓
Deployment
```

---

# Week 12 — Infrastructure as Code

### Topic

Infrastructure

### Core Tool

* Terraform

### Objective

Provision infrastructure programmatically and reproducibly.

### Suggested Task

> Deploy infrastructure using Terraform.

Key concepts:

* infrastructure as code
* reproducible environments
* automated provisioning
* configuration management

---

# Final Capstone

## End-to-End ML System

The final capstone combines the concepts from the entire course into one production-oriented machine learning system.

### Objective

> Deliver and present a production-ready MLOps solution.

The final system should demonstrate the complete lifecycle:

```text
                 ┌───────────────┐
                 │    Raw Data   │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │   ETL/Data    │
                 │   Pipeline    │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │ Feature Store │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │ Model Training│
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │    MLflow     │
                 │    Registry   │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │ Model Serving │
                 │   / API       │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │    Docker     │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │  Monitoring   │
                 │ Prometheus /  │
                 │   Grafana     │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │   Automated   │
                 │   Retraining  │
                 └───────────────┘
```

---

# Tools Covered

Throughout the course, the following technologies are introduced:

### Machine Learning

* Python
* Scikit-learn

### Deep Learning

* PyTorch
* Hugging Face

### APIs

* FastAPI
* Pydantic

### Containers

* Docker

### Data Engineering

* Python ETL
* PostgreSQL / databases

### Data Versioning

* DVC

### Experiment Tracking

* MLflow
* MLflow Model Registry

### Distributed Computing

* Ray Core
* Ray Train
* Ray Serve

### Feature Management

* Feast

### Monitoring

* Prometheus
* Grafana

### Automation

* GitHub Actions

### Infrastructure

* Terraform

---

# MLOps Principles

The course focuses on several important principles.

## Reproducibility

The same code, data, preprocessing, and model versions should be reproducible.

## No Data Leakage

Information from validation or test data should not influence model training or preprocessing decisions.

## Version Everything Important

Datasets, models, preprocessing artifacts, configurations, and experiments should be traceable.

## Automate Repetitive Processes

Manual model training and deployment processes should gradually be replaced with automated pipelines.

## Monitor Production Systems

A deployed model is not finished when it goes into production. Its data, predictions, performance, and infrastructure need to be monitored.

## Continuous Improvement

Production systems should support retraining and redeployment when new data or model improvements become available.

---

# Current Progress

| Phase                              | Status         |
| ---------------------------------- | -------------- |
| Week 1 — Leakage-proof ML Pipeline | ✅ Completed    |
| Week 2 — Deep Learning Pipeline    | 🔄 In Progress  |
| Week 3 — Production API            | ⬜              |
| Week 4 — Docker                    | ⬜              |
| Week 5 — ETL Pipeline              | ⬜              |
| Week 6 — DVC                       | ⬜              |
| Week 7 — MLflow                    | ⬜              |
| Week 8 — Distributed ML            | ⬜              |
| Week 9 — Feature Store             | ⬜              |
| Week 10 — Monitoring               | ⬜              |
| Week 11 — Continuous Retraining    | ⬜              |
| Week 12 — Terraform                | ⬜              |
| Final Capstone                     | ⬜              |

> Update the status table as each phase is completed.

---

# Environment

The project uses Python and a project-level virtual environment managed with `uv`.

Typical setup:

```bash
uv sync
```

Run notebooks using the project's Python environment/kernel.

Generated artifacts are stored under:

```text
artifacts/
```

and should not be committed to Git unless specifically required.

---

# Key Learning Goal

The main goal of this course is to move beyond simply **training a machine learning model**.

The progression is:

```text
"I can train a model."
          ↓
"I can build a reliable ML pipeline."
          ↓
"I can expose the model through an API."
          ↓
"I can package it with Docker."
          ↓
"I can automate the data pipeline."
          ↓
"I can version data and models."
          ↓
"I can track experiments."
          ↓
"I can scale training."
          ↓
"I can manage production features."
          ↓
"I can monitor the system."
          ↓
"I can automate retraining."
          ↓
"I can provision infrastructure."
          ↓
"I can build an end-to-end MLOps system."
```

---

# Repository Goal

By the end of the Qafza MLOps course, this repository should demonstrate a complete progression from a locally trained machine learning model to a reproducible, deployable, monitored, and automated ML system.

The final capstone will bring the individual components together into a production-oriented architecture.

---

## Author

**Lana Ismail**

Data & AI | Machine Learning | MLOps

This repository documents my practical learning journey through the Qafza MLOps course.
