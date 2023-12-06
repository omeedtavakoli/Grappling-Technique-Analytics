import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataset_path = '/path/to/your/dataset.csv'  # Replace with the correct path
data = pd.read_csv(dataset_path)

# Display the first five rows of the dataset for a preliminary check
print("First five rows of the dataset:")
print(data.head())

# Generate and display a bar chart showing the distribution of positions
position_counts = data['Position'].value_counts()
position_counts.plot(kind='bar')
plt.title('Position Counts')
plt.show()  # Shows the plot with position counts

# Check for missing values in the dataset and print the count per column
print("Missing values per column:")
print(data.isnull().sum())

# Provide a statistical summary of the dataset
print("Statistical summary:")
print(data.describe(include='all'))

# Output unique values in categorical columns
print("Unique values in 'Type':", data['Type'].unique())
print("Unique values in 'Position':", data['Position'].unique())
print("Unique values in 'Origin':", data['Origin'].unique())

# Encoding categorical variables using Label Encoding
label_encoder = LabelEncoder()
for column in ['Type', 'Origin']:
    data[column] = label_encoder.fit_transform(data[column])

# Define the feature set (X) and the target variable (y)
X = data.drop(['Position', 'Name'], axis=1)  # Excluding 'Name' as it's a unique identifier
y = data['Position']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Train the Random Forest Classifier with the training data
model.fit(X_train, y_train)

# Perform cross-validation to assess the model's stability and performance
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", cv_scores)

# Use the trained model to make predictions on the test dataset
y_pred = model.predict(X_test)

# Evaluate the model's performance and print the accuracy and a detailed classification report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate and print the feature importance as determined by the model
importances = model.feature_importances_
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, importances))
# Print the feature importance in descending order
for name, importance in sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True):
    print(f"{name}: {importance}")

# Enhanced Visualizations
# Technique distribution across different styles
for style in data['Origin'].unique():
    subset = data[data['Origin'] == style]
    technique_counts = subset['Type'].value_counts()
    technique_counts.plot(kind='bar', title=f'Technique Distribution in Style {style}')
    plt.show()
