# Student Pass/Fail Prediction System

A Streamlit-based web application for predicting student pass/fail outcomes using an ensemble machine learning model.

## Features

✨ **Single Student Prediction**
- Input individual student academic metrics
- Get instant pass/fail prediction with confidence scores
- View probability breakdown for both outcomes

📊 **Batch Prediction**
- Upload CSV files with multiple students
- Process predictions for all students at once
- Download results with probabilities and confidence scores
- View summary statistics

📈 **Model Information**
- Detailed model architecture explanation
- Input feature descriptions
- Tips for accurate predictions
- Prediction methodology

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd c:\Users\Solowyse\Desktop\FinalYear
   ```

2. **Activate the virtual environment:**
   ```bash
   # On Windows PowerShell
   .\venv\Scripts\Activate.ps1
   
   # On Windows Command Prompt
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install/Update required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

### Single Student Prediction

1. Navigate to the **Single Student Prediction** tab
2. Enter the student's academic metrics:
   - **CGPA:** Cumulative Grade Point Average (0.0-4.0)
   - **WASSCE Grade:** WASSCE examination grade
   - **Current Semester:** Semester number (1-8)
   - **Semester GPA:** Current semester GPA (0.0-4.0)
3. Click **"🔮 Predict Pass/Fail"**
4. View the prediction result with:
   - Pass/Fail status
   - Pass probability percentage
   - Fail probability percentage
   - Confidence scores
   - Input summary

### Batch Prediction

1. Navigate to the **Batch Prediction** tab
2. Prepare a CSV file with the following columns:
   - `Gender`
   - `CGPA`
   - `First_Year_CGPA`
   - `Wassce Grade`
   - `Programme of Study`
3. Upload the CSV file
4. Click **"🔮 Predict for All Students"**
5. Review the predictions and download results as CSV

#### Sample CSV Format

```csv
Gender,CGPA,First_Year_CGPA,Wassce Grade,Programme of Study
Male,3.5,3.2,85.0,Engineering
Female,2.8,2.5,72.5,Science
Male,3.2,3.0,78.0,Business
Female,1.9,1.8,65.0,Arts
Male,3.8,3.6,92.0,Engineering
```

## Model Details

### Architecture
- **Model Type:** Voting Classifier (Ensemble)
- **Base Estimators:**
  - Random Forest Classifier
  - Support Vector Classifier (SVC)
  - Gradient Boosting Classifier

### Preprocessing
- StandardScaler for numerical feature normalization
- OneHotEncoder for categorical features
- Imputation for handling missing values
- SMOTE for class balancing during training

### Input Features
- **Gender:** Student's gender (Male/Female)
- **CGPA:** Cumulative Grade Point Average (0.0-4.0)
- **First_Year_CGPA:** CGPA from first year (0.0-4.0)
- **Wassce Grade:** WASSCE examination grade score
- **Programme of Study:** Student's academic programme

### Target Variable
- **Pass:** CGPA ≥ 2.0
- **Fail:** CGPA < 2.0

## File Structure

```
FinalYear/
├── app.py                           # Streamlit application
├── requirements.txt                 # Python dependencies
├── models/
│   ├── pass_fail_voting_pipeline.pkl    # Trained voting classifier
│   ├── preprocessor.pkl                 # Data preprocessor
│   ├── random_forest_tuned.pkl          # RF model component
│   ├── svc_tuned.pkl                    # SVC model component
│   └── gradient_boosting_tuned.pkl      # GB model component
├── dataset/                         # Dataset files
├── utils/
│   └── Pass_Fail_VC.ipynb           # Model training notebook
├── images/                          # Generated visualizations
└── venv/                            # Virtual environment
```

## Prediction Output

The application provides:

1. **Prediction Result**
   - Clear PASS/FAIL status with emoji indicators
   - Color-coded cards (green for PASS, red for FAIL)

2. **Probability Scores**
   - Pass probability percentage
   - Fail probability percentage
   - Confidence level
   - Visual progress bars

3. **Detailed Analysis**
   - Tabular breakdown of probabilities
   - Input summary with all provided values

## Tips for Better Predictions

✅ **DO:**
- Ensure all values are accurate and current
- Use valid ranges for each metric
- Include all required fields
- Verify CGPA is on the 0-4.0 scale
- Provide semester information accurately

❌ **DON'T:**
- Leave any fields empty
- Use invalid value ranges
- Mix different GPA scales
- Submit incomplete student records

## Troubleshooting

### Models not found
- Ensure `models/` directory contains all `.pkl` files
- Check file permissions

### Invalid input values
- Verify CGPA is between 0.0 and 4.0
- Ensure Wassce Grade is a valid number
- Confirm Semester is between 1-8
- Check that GPA is between 0.0 and 4.0

### CSV upload errors
- Verify column names match exactly: `CGPA`, `Wassce Grade`, `Semester`, `GPA`
- Ensure no missing values in the CSV
- Check file encoding (UTF-8 recommended)

### Streamlit connection issues
- Check if port 8501 is available
- Try running with: `streamlit run app.py --server.port 8502`

## Performance Metrics

Based on the training data:
- The ensemble model combines strengths of three classifiers
- Voting mechanism improves prediction robustness
- SMOTE addresses class imbalance
- Model evaluated using cross-validation

## Future Enhancements

- Add confidence intervals
- Include feature importance visualization
- Support for additional student metrics
- Historical prediction tracking
- Export to PDF reports
- Real-time model retraining options

## Technical Stack

- **Frontend:** Streamlit
- **Backend:** Python, scikit-learn
- **Data Processing:** pandas, numpy
- **Serialization:** pickle
- **Visualization:** matplotlib, seaborn

## Support

For issues or questions:
1. Check the Model Information tab in the app
2. Review the troubleshooting section above
3. Verify input data format and values
4. Check console output for error messages

## License

This project is for educational purposes.

## Created with ❤️

Built for academic performance prediction and student support systems.
