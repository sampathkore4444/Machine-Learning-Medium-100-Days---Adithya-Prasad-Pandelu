import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# Sample data
data = {
    "item_price": [100, 150, 200, 250, 300],
    "quantity_sold": [1, 3, 2, 5, 4],
    "store_location": ["New Delhi", "Mumbai", "Chennai", "Kolkata", "New Delhi"],
}
df = pd.DataFrame(data)
print("Original Data:")
print(df)

# Normalization using MinMaxScaler
scaler = MinMaxScaler()
df[["item_price", "quantity_sold"]] = scaler.fit_transform(
    df[["item_price", "quantity_sold"]]
)
print("Normalized data for columns (item_price and quantity_sold) using MinMaxScaler:")
print(df)

# Standardization using StandardScaler
scaler = StandardScaler()
df[["item_price_std", "quantity_sold_std"]] = scaler.fit_transform(
    df[["item_price", "quantity_sold"]]
)

print(
    "Standardized data for columns (item_price and quanity_sold) using StandardScaeler:"
)
print(df)

# Encoding categorical data using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
location_encoded = encoder.fit_transform(df[["store_location"]])
print("Encoded data:")
print(location_encoded)
# print(encoder.get_feature_names_out(["store_location"]))
location_df = pd.DataFrame(
    location_encoded, columns=encoder.get_feature_names_out(["store_location"])
)
print(location_df)

# concatenate encoded columns with original data
df = pd.concat([df, location_df], axis=1)
print(df)

# drop original categorical column
df = df.drop("store_location", axis=1)
print(df)
