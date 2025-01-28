import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import xverse
from xverse.transformer import WOE
from woe.feature_process import proc_woe_continuous, proc_woe_discrete, calulate_iv  # Fixed typo

# Load your data
df = pd.read_csv("../data/raw/data.csv")

# Step 1: Create Aggregate Features
df['TotalTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('sum')
df['AverageTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('mean')
df['TransactionCount'] = df.groupby('CustomerId')['TransactionId'].transform('count')
df['StdTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('std')

# Step 2: Extract Time-based Features
df['TransactionHour'] = pd.to_datetime(df['TransactionStartTime']).dt.hour
df['TransactionDay'] = pd.to_datetime(df['TransactionStartTime']).dt.day
df['TransactionMonth'] = pd.to_datetime(df['TransactionStartTime']).dt.month
df['TransactionYear'] = pd.to_datetime(df['TransactionStartTime']).dt.year

# Step 3: Encode Categorical Variables
# One-Hot Encoding for 'CountryCode' and 'ProductCategory'
df = pd.get_dummies(df, columns=['CountryCode', 'ProductCategory'], drop_first=True)

# Label Encoding for 'ChannelId'
le = LabelEncoder()
df['ChannelId'] = le.fit_transform(df['ChannelId'])

# Step 4: Handle Missing Values
# Imputation for 'Amount' with mean
df['Amount'] = df['Amount'].fillna(df['Amount'].mean())

# Step 5: Normalize/Standardize Numerical Features
# Normalization (scaling to [0, 1] range)
scaler = MinMaxScaler()
df[['Amount']] = scaler.fit_transform(df[['Amount']])

# Alternatively, Standardization (scaling to mean 0 and std 1)
# scaler = StandardScaler()
# df[['Amount']] = scaler.fit_transform(df[['Amount']])

# Step 6: Feature Engineering with xverse (optional)
# If you choose to use xverse, this step will integrate it into the process
# fe = xverse.FeatureEngineering()  # Ensure you have xverse installed and running properly
# df = fe.fit_transform(df)

# Step 7: Calculate WOE and IV for Features
# Set the global threshold and minimum sample parameters
global_gt = 10  # Example threshold for binning
min_sample = 5  # Example minimum samples per bin

# Add 'target' column explicitly for WOE
df['target'] = df['FraudResult']

# Calculate WOE for continuous features (e.g., 'Amount')
woe_continuous = proc_woe_continuous(df, 'target', ['Amount'], global_gt, min_sample)

# Calculate WOE for discrete features (e.g., 'ProductCategory')
woe_discrete = proc_woe_discrete(df, 'target', ['ProductCategory'], global_gt, min_sample)

# Calculate Information Value (IV)
iv_continuous = calulate_iv(df, 'target', ['Amount'])
iv_discrete = calulate_iv(df, 'target', ['ProductCategory'])

# Final DataFrame with engineered features
print("Final Dataframe with Engineered Features:")
print(df.head())

# Print WOE and IV results
print("WOE for Continuous Features:")
print(woe_continuous.head())

print("WOE for Discrete Features:")
print(woe_discrete.head())

print(f"IV for Continuous Feature (Amount): {iv_continuous}")
print(f"IV for Discrete Feature (ProductCategory): {iv_discrete}")
