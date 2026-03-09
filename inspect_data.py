import pandas as pd
df = pd.read_csv("data/train.csv")
print("First rows:")
print(df.head())
print("\nClass distribution:")
print(df['diagnosis'].value_counts())