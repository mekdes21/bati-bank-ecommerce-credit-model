# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("../data/raw/data.csv")

# Display the first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Check the structure of the dataset
print("\nDataset Structure:")
print(data.info())

# Check for the number of rows and columns
print(f"\nRows: {data.shape[0]}, Columns: {data.shape[1]}")

# ------------------- 2. Summary Statistics -------------------
# Summary statistics for numerical features
print("\nSummary Statistics for Numerical Features:")
print(data.describe())

# Check for unique values in categorical features
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\nUnique values in {col}: {data[col].nunique()}")

# ------------------- 3. Distribution of Numerical Features -------------------
# Plot histograms for numerical features
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_columns].hist(figsize=(15, 10), bins=20, edgecolor='black')
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.show()

# Plot individual feature distributions
for col in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data[col], fill=True, label=col)
    plt.title(f"Distribution of {col}")
    plt.show()


# ------------------- 4. Distribution of Categorical Features -------------------
# Plot bar charts for categorical features
# Limit to top 10 most frequent categories
top_n = 10  # Adjust as needed
for col in categorical_columns:
    # Get value counts for categories
    value_counts = data[col].value_counts().head(top_n)
    
    # Create a bar plot using matplotlib
    plt.figure(figsize=(10, 6))  # Increase figure size for better spacing
    plt.bar(value_counts.index, value_counts.values)
    plt.title(f"Distribution of {col}")
    
    # Adjust x-axis ticks and spacing
    plt.xticks(rotation=45, ha='right')  # Rotate labels and adjust horizontal alignment
    plt.subplots_adjust(bottom=0.2)  # Increase bottom margin to avoid overlap
    
    # Show plot
    plt.show()


# ------------------- 5. Correlation Analysis -------------------
# Select only numerical columns for correlation calculation
numerical_data = data.select_dtypes(include=['number'])

# Calculate the correlation matrix
corr_matrix = numerical_data.corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Identify highly correlated pairs (threshold = 0.7)
high_corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
high_corr_pairs = high_corr_pairs[high_corr_pairs != 1]  # Exclude self-correlation
print("Highly Correlated Pairs (Threshold > 0.7):")
print(high_corr_pairs[high_corr_pairs > 0.7])

# ------------------- 6. Identifying Missing Values -------------------
# Separate numerical and categorical columns
numerical_columns = data.select_dtypes(include=['number']).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Check for missing values
missing_values = data.isnull().sum()
missing_percentage = (missing_values / len(data)) * 100
missing_data = pd.DataFrame({"Missing Count": missing_values, "Percentage": missing_percentage})
print(missing_data[missing_data["Missing Count"] > 0])

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

# Imputation Strategies

# Fill missing numerical data with the median
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())

# Fill categorical missing values with the mode
for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Check if there are any missing values left
missing_values_after_imputation = data.isnull().sum()
print(f"Missing values after imputation:\n{missing_values_after_imputation}")

print("\nMissing values imputed.")

# ------------------- 7. Outlier Detection -------------------
# Box plots for numerical features to detect outliers
print("\nPlotting box plots for numerical features to detect outliers:")
# Detect and cap outliers using IQR
for col in numerical_columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap outliers
    data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

# Check for remaining outliers
for col in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=data[col], showfliers=True)
    plt.title(f"Box Plot After Outlier Capping: {col}")
    plt.xlabel(col)
    plt.show()


print("\nOutliers handled.")

# ------------------- 8. Save Cleaned Data -------------------
# After completing the EDA, save the cleaned dataset for use in modeling
data.to_csv("../data/processed/cleaned_data.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_data.csv'.")
