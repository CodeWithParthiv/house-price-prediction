# House Price Prediction Web Application

This is a machine learning-based web application that predicts house prices based on various features such as area, number of bedrooms, location, and more.

## Features

- Predicts house prices using a Random Forest model
- Modern and responsive web interface
- Real-time predictions
- Handles both numerical and categorical features

## Setup Instructions

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Run the Flask application:
```bash
python app.py
```

4. Open your web browser and navigate to `http://localhost:5000`

## Input Features

The model takes into account the following features:
- Area (in square feet)
- Number of bedrooms
- Number of bathrooms
- Number of stories
- Main road access (yes/no)
- Guest room (yes/no)
- Basement (yes/no)
- Hot water heating (yes/no)
- Air conditioning (yes/no)
- Number of parking spaces
- Preferred area (yes/no)
- Furnishing status (furnished/semi-furnished/unfurnished)

## Model Details

- Algorithm: Random Forest Regressor
- Features are scaled using StandardScaler
- Categorical variables are encoded appropriately
- The model is saved using joblib for efficient loading

## Technologies Used

- Python
- Flask
- scikit-learn
- pandas
- numpy
- Bootstrap
- jQuery 