import os
import pandas as pd
import pytest
from utils import save_feedback

# Define a temporary file path for testing
TEST_FEEDBACK_FILE = 'test_feedback.csv'

def test_save_feedback_creates_file():
    """Tests that a new feedback file is created if it doesn't exist."""
    # Cleanup before test
    if os.path.exists(TEST_FEEDBACK_FILE):
        os.remove(TEST_FEEDBACK_FILE)

    assert save_feedback("Test User", "test@example.com", "Great app!", TEST_FEEDBACK_FILE) == True
    
    assert os.path.exists(TEST_FEEDBACK_FILE)
    df = pd.read_csv(TEST_FEEDBACK_FILE)
    assert len(df) == 1
    assert df.iloc[0]['Name'] == "Test User"
    assert df.iloc[0]['Comments'] == "Great app!"
    
    # Cleanup after test
    os.remove(TEST_FEEDBACK_FILE)

def test_save_feedback_appends_to_existing_file():
    """Tests that feedback is appended to an existing file."""
    # Create an initial file
    initial_data = {'Timestamp': ['2023-01-01'], 'Name': ['Initial'], 'Email': ['initial@test.com'], 'Comments': ['Initial comment']}
    pd.DataFrame(initial_data).to_csv(TEST_FEEDBACK_FILE, index=False)

    assert save_feedback("Appender", "appender@example.com", "Appending feedback", TEST_FEEDBACK_FILE) == True
    
    df = pd.read_csv(TEST_FEEDBACK_FILE)
    assert len(df) == 2
    assert df.iloc[1]['Name'] == "Appender"
    
    # Cleanup after test
    os.remove(TEST_FEEDBACK_FILE)

def test_save_feedback_handles_exceptions(mocker):
    """Tests that the function returns False if an exception occurs."""
    # Mock os.path.isfile to raise an exception
    mocker.patch('pandas.DataFrame.to_csv', side_effect=IOError("Disk full"))
    
    assert save_feedback("Error User", "err@example.com", "This should fail", TEST_FEEDBACK_FILE) == False
