import pandas as pd

# Read Excel file
df = pd.read_excel("data\\Ports\\ports.xlsx")

# Show first 5 rows
print(df.head())