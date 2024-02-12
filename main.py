import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


def clean_data(input_file_path, cleaned_data_output_path):
  # Load data from Excel
  df = pd.read_excel(input_file_path)

  # Drop unwanted columns
  unwanted_columns = [
      'Name', 'Parental Backgrond', 'Good in marks and subjects', 'Unnamed: 18'
  ]
  df_cleaned = df.drop(columns=unwanted_columns)
  df_cleaned.dropna(subset=['Field'], inplace=True)

  # Save cleaned data to CSV
  df_cleaned.to_csv(cleaned_data_output_path, index=False)
  print("Data Cleaning complete. Cleaned data saved.")
  print("Column Names:", df_cleaned.columns)


def train_linear_regression_model(cleaned_data_file):
  # Load cleaned data
  cleaned_data = pd.read_csv(cleaned_data_file)

  # Encode categorical features
  label_encoder = LabelEncoder()
  for column in cleaned_data.columns:
    if cleaned_data[column].dtype == 'object':
      cleaned_data[column] = label_encoder.fit_transform(cleaned_data[column])

  # Train Linear Regression model
  X = cleaned_data.drop(columns=['Field'])
  y = cleaned_data['Field']

  model = LinearRegression()
  model.fit(X, y)

  return model, label_encoder


def predict_field(model, new_data, label_encoder):
  # Columns to use for prediction
  all_columns_except_field = [
      'Sr. No.', 'History', 'Geography', 'Political Science', 'Economics',
      'Maths', 'Physics', 'Chemistry', 'Biology', 'Accounts',
      'Physical Education', 'Sports', 'Indoor sports', 'Art and Craft',
      'Music', 'Dance'
  ]

  # Ensure 'new_data' has all necessary columns for prediction
  missing_columns = list(set(all_columns_except_field) - set(new_data.columns))
  for col in missing_columns:
    new_data[col] = 0

  # Reorder columns to match expected order
  new_data = new_data[all_columns_except_field]

  # Encode categorical features in new data
  for column in new_data.columns:
    if new_data[column].dtype == 'object':
      new_data[column] = label_encoder.transform(new_data[column])

  predictions = model.predict(new_data)
  inverse_transformed_predictions = label_encoder.inverse_transform(
      predictions.round().astype(int))

  return inverse_transformed_predictions


input_file_path = 'project.xlsx'  # Replace with your actual file path
cleaned_data_output_path = 'cleaned_data.csv'  # Replace with your desired output path
clean_data(input_file_path, cleaned_data_output_path)
print("Data Cleaning complete. Cleaned data saved.")

target_variable = 'Field'  # Replace with your target variable
model, label_encoder = train_linear_regression_model(cleaned_data_output_path)
print("Linear Regression Model trained.")

# Generate random scores for each subject
new_data = pd.DataFrame({
    'Sr. No.': [1],
    'History': [3],  # Replace with the actual value
    'Geography': [4],  # Replace with the actual value
    'Political Science': [5],
    'Economics': [8],
    'Maths': [20],
    'Physics': [12],
    'Chemistry': [10],
    'Biology': [30],
    'Accounts': [45],
    'Physical Education': [45],
    'Sports': [14],
    'Indoor sports': [45],
    'Art and Craft': [78],
    'Music': [20],
    'Dance': [91],
})

# Identifying the subject with the highest score
subject_scores = new_data.iloc[0].to_dict()
highest_score_subject = max(subject_scores, key=subject_scores.get)

# Creating new_data with only the highest score subject
new_data = pd.DataFrame({
    'Sr. No.': [1],
    highest_score_subject: [subject_scores[highest_score_subject]]
})

predictions = predict_field(model, new_data, label_encoder)
predicted_field = predictions[0]
print("\nPredicted Field:", predicted_field)
