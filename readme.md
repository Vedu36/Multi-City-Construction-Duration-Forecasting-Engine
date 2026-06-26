# V13 - Static Construction Duration Prediction Model

## Overview

V13 is a production-ready static construction duration forecasting model designed to predict the duration of each construction stage and the overall project timeline for residential home-building projects.

The model was developed using historical construction data and optimized to provide accurate stage-level predictions while maintaining scalability across multiple cities and divisions.

---

## Structure

```text
├── V13.ipynb                         # Local model training notebook
├── run_inference_v13.py              # Local inference pipeline
├── requirements.txt                  # Python dependencies
├── output_v13/                       # Trained model artifacts
├── predictions_v13.csv               # Sample prediction output
├── SQL_Quary_Final_12_5_26.sql       # The SQL Quary  

├── fabric/
│   ├── construction_pipeline_v13_fabric.ipynb     # Microsoft Fabric training notebook
│   └── run_inference_v13_fabric.ipynb             # Microsoft Fabric inference notebook
└── README.md
```

---

## Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Model Training

The primary training workflow is contained within:

```text
V13.ipynb
```

This notebook:

- Loads historical construction data
- Performs data cleaning and preprocessing
- Generates leakage-free features
- Creates city-level and global fallback models
- Trains stage-specific forecasting models
- Saves trained artifacts to the output directory

### Training Output

After training completes, model files are stored in:

```text
output_v13/
```

---

## Running Inference

The inference pipeline generates duration predictions for active construction projects using the trained V13 models.

### Command

```bash
python run_inference_v13.py \
    --fact Ongoing_Actual.csv \
    --stage Ongoing_Stage.csv \
    --models ./output_v13 \
    --out predictions_v13.csv
```

### Optional Arguments

Disable log-space transformations:

```bash
python run_inference_v13.py \
    --fact Ongoing_Actual.csv \
    --stage Ongoing_Stage.csv \
    --models ./output_v13 \
    --out predictions_v13.csv \
    --no-log-space
```

### Parameters

| Parameter | Description |
|------------|------------|
| --fact | Active construction jobs dataset |
| --stage | Construction stage tracking dataset |
| --models | Directory containing trained model artifacts |
| --out | Output prediction file |
| --no-log-space | Disable log-space prediction transformation |

### Output

```text
predictions_v13.csv
```

Contains:

- Predicted stage durations
- Predicted stage completion dates
- ML forecast metrics
- FourMonthAvg baseline comparisons

---

## Microsoft Fabric Support

This repository includes Microsoft Fabric implementations for enterprise deployment.

### Fabric Training Notebook

```text
fabric/construction_pipeline_v13_fabric.ipynb
```

Used for:

- Model training in Fabric
- Lakehouse integration
- Enterprise-scale processing
- Model artifact generation

### Fabric Inference Notebook

```text
fabric/run_inference_v13_fabric.ipynb
```

Used for:

- Batch prediction generation
- Lakehouse data ingestion
- Prediction output storage
- Operational forecasting workflows

---

## Model Artifacts

Generated models are stored in:

```text
output_v13/
```

Artifacts include:

- Stage-specific models
- City-specific models
- Global fallback models
- Feature configuration files
- Encoding metadata

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- LightGBM
- XGBoost
- Microsoft Fabric
- Synapse Analytics
- Lakehouse Architecture

## Key Features

### City-Specific Modeling

- Separate machine learning models are trained for each city whenever sufficient historical data is available.
- Cities with limited job volume are grouped with cities that exhibit similar construction behavior.
- Cities with very low job counts utilize a global fallback model to ensure prediction stability and reliability.

### Stage-Based Forecasting

The model predicts the duration of the following construction stages:

1. Foundation
2. Frame
3. Cornice
4. Mechanicals
5. Sheetrock
6. Trim
7. Interior

Each stage is modeled independently, allowing stage-specific patterns and delays to be learned more effectively.

### Sequential Stage Chaining

Predicted completion dates are propagated through the construction lifecycle:

Foundation → Frame → Cornice → Mechanicals → Sheetrock → Trim → Interior

The predicted completion date of one stage becomes the starting point for the next stage unless an actual completion date already exists.

---

## Model Architecture

### Per-Stage Models

Each construction stage has its own dedicated machine learning model.

### Per-City Routing

During inference:

- If a city-specific model exists, the prediction is generated using that model.
- If the city belongs to a bundled city group, the grouped model is used.
- If insufficient historical data exists, a global model is used.

This hierarchical routing approach improves prediction accuracy while maintaining coverage for all cities.

---

## Features Used

The model utilizes only historical and operational data available before the start of a stage.

### Calendar & Seasonal Features

- Month
- Quarter
- Season
- Holiday proximity
- Winter indicators
- Cyclical date encodings

### Historical Construction Performance

- Rolling averages of recent jobs
- Community-level performance trends
- City-level performance trends
- Lag features
- Expanding mean and standard deviation metrics

### Construction Delay Features

- Permit delays
- HVAC delays
- Plumbing delays
- Engineering delays
- Utility-related delays

Missing delay information is imputed using city-level historical averages.

### Project Attributes

- City
- Community
- Floor Plan
- Construction characteristics
- Encoded categorical variables

---

## Data Leakage Prevention

V13 was designed with strict leakage prevention:

- Only historical information is used.
- Rolling and expanding statistics use past records only.
- Future project information is never included during training or inference.
- Actual stage completion dates always override predicted dates when available.

---

## Baseline Comparison

### FourMonthAvg Baseline

A traditional four-month moving average approach is used as a benchmark.

V13 compares:

- ML Prediction vs Actual
- FourMonthAvg vs Actual

This allows continuous evaluation of machine learning improvements over historical forecasting methods.

---

## Limitations

V13 is a static prediction system.

The model does not currently use:

- Real-time weather forecasts
- Weather alerts
- Federal emergency alerts
- Live operational disruptions
- Dynamic construction events

Seasonal variables such as month, holiday indicators, and winter flags act only as indirect proxies for weather conditions.

---

## Performance Characteristics

- Optimized for large-scale construction forecasting.
- Supports 44,000+ historical construction records.
- Handles city-specific behavior through model specialization.
- Provides stage-level and project-level timeline predictions.
- Designed for deployment within enterprise forecasting workflows.

---

## Future Evolution

V13 serves as the foundation for V14.

V14 extends the static forecasting framework by introducing a dynamic weather residual layer capable of adjusting stage predictions based on:

- Weather forecasts
- Severe weather alerts
- Environmental conditions
- Real-time external factors

This evolution transforms the forecasting system from a static prediction engine into a dynamic construction scheduling platform.