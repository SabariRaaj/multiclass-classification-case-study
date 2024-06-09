# Activity Classification App

This project is an activity classification application built with Streamlit, TensorFlow, and XGBoost. The application predicts the type of physical activity based on input features.

## Table of Contents

- Overview
- Features
- Requirements
- Installation
- Usage

## Overview

The Activity Classification App allows users to input activity data and classify it into different types of activities using machine learning models. The models are pre-trained and saved as `.joblib` files.

The classification models in this project are designed to categorize the following types of physical activities:
1: Working at Computer
2: Standing Up, Walking and Going up\down stairs
3: Standing
4: Walking
5: Going Up\Down Stairs
6: Walking and Talking with Someone
7: Talking while Standing

Each activity is characterized by the inputs of x acceleration, y acceleration and z acceleration from wearable accelerometer, which are used by the models to make accurate predictions.


## Features

- User-friendly web interface built with Streamlit
- Classification using pre-trained XGBoost and TensorFlow models
- Visualizations of the input data

## Requirements

- Python 3.7 or higher
- Streamlit
- TensorFlow
- XGBoost
- Joblib
- Pandas
- Numpy
- Matplotlib

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Ensure that the `models/` directory contains the pre-trained model files (`xgb_model.joblib` and `activity_model.h5`).

## Usage

To run the application, use the Streamlit command:

```bash
streamlit run activity_classification_app.py
