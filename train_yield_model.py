import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_and_save_model():
    """
    Loads the yield data, trains a RandomForestRegressor model,
    and saves it to the 'models' directory.
    """
    print("Starting model training process...")

    # Define paths
    data_path = 'yield_df.csv'
    models_dir = 'models'
    model_path = os.path.join(models_dir, 'yield_regressor.joblib')

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at '{data_path}'")
        return

    print(f"Loading data from '{data_path}'...")
    # Load the dataset
    df = pd.read_csv(data_path)

    # Preprocessing
    df.rename(columns={'hg/ha_yield': 'yield'}, inplace=True)
    df = pd.get_dummies(df, columns=['Area', 'Item'], drop_first=True)

    X = df.drop('yield', axis=1)
    y = df['yield']

    # Splitting data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data preprocessed and split.")

    # Model training
    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: '{models_dir}'")

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Model saved successfully to '{model_path}'")

if __name__ == '__main__':
    train_and_save_model()
