from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing(as_frame=True)

df = pd.DataFrame(data.data, columns=data.feature_names)
df['price'] = data.target
print('shape of the dataset:', df.shape)
print(df.head())
print(df.describe())
print(df.info())

