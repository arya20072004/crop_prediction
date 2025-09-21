import pandas as pd
from fpdf import FPDF
import datetime
import os
import streamlit as st

def save_feedback(name, email, comments, feedback_file='feedback.csv'):
    """
    Saves user feedback to a CSV file.
    Uses os.path.join for cross-platform compatibility.
    """
    try:
        feedback_dir = os.path.dirname(feedback_file)
        if feedback_dir and not os.path.exists(feedback_dir):
            os.makedirs(feedback_dir)

        new_feedback = pd.DataFrame({
            'Timestamp': [datetime.datetime.now()],
            'Name': [name],
            'Email': [email],
            'Comments': [comments]
        })

        if not os.path.isfile(feedback_file):
            new_feedback.to_csv(feedback_file, index=False)
        else:
            new_feedback.to_csv(feedback_file, mode='a', header=False, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return False

def create_downloadable_pdf(prediction_data):
    """
    Generates a downloadable PDF report from prediction data.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)

    # Title
    pdf.cell(0, 10, 'AI Agriculture Assistant Report', 0, 1, 'C')
    pdf.ln(10)

    # Report Info
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(5)

    # Content
    for key, value in prediction_data.items():
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(50, 10, f"{key}:", 0, 0)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, str(value), 0, 1)
        pdf.ln(2)

    # Convert PDF to bytes for download
    return pdf.output(dest='S').encode('latin-1')
