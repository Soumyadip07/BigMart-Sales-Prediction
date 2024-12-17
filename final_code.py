import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

train_data=pd.read_csv("train_v9rqX0R.csv")
test_data=pd.read_csv("test_AbJTz2l.csv")
for dataset in [train_data, test_data]:
    dataset['Item_Weight'] = dataset['Item_Weight'].fillna(dataset['Item_Weight'].median())
train_null_count = train_data['Item_Weight'].isnull().sum()
test_null_count = test_data['Item_Weight'].isnull().sum()
outlet_columns = [col for col in train_data.columns if 'Outlet' in col]
for data in [train_data, test_data]:
    data_missing_size = data[data['Outlet_Size'].isnull()]
    data_non_missing_size = data[data['Outlet_Size'].notnull()]
    features = ['Outlet_Location_Type', 'Outlet_Type']
    encoded_non_missing = pd.get_dummies(data_non_missing_size[features], drop_first=True)
    encoded_missing = pd.get_dummies(data_missing_size[features], drop_first=True)
    encoded_non_missing, encoded_missing = encoded_non_missing.align(encoded_missing, join='left', axis=1, fill_value=0)
    y = data_non_missing_size['Outlet_Size']
    X_train, X_test, y_train, y_test = train_test_split(encoded_non_missing, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    predicted_sizes = rf_model.predict(encoded_missing)
    data.loc[data['Outlet_Size'].isnull(), 'Outlet_Size'] = predicted_sizes

plt.figure(figsize=(15, 12))
columns_to_plot = ['Item_Weight','Item_Visibility','Item_MRP']
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, 3, i)
    plt.boxplot(train_data[column].dropna(), vert=False)
    plt.title(f"Train Data: Boxplot of {column}")
    plt.xlabel(column)
for i, column in enumerate(columns_to_plot, 4):
    plt.subplot(2, 3, i)
    plt.boxplot(test_data[column].dropna(), vert=False)
    plt.title(f"Test Data: Boxplot of {column}")
    plt.xlabel(column)
plt.tight_layout()
plt.show()
Q1 = test_data['Item_Visibility'].quantile(0.25)
Q3 = test_data['Item_Visibility'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
test_data['Item_Visibility'] = test_data['Item_Visibility'].apply(
    lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
)
for df in [train_data, test_data]:
    median_visibility = df['Item_Visibility'].median()
    df['Item_Visibility'] = df['Item_Visibility'].replace(0, median_visibility)
for dataset in [train_data,test_data]:
    dataset['Item_Fat_Content'] = dataset['Item_Fat_Content'].replace({
        'Low Fat': 'low_fat',
        'LF': 'low_fat',
        'low fat': 'low_fat',
        'Regular': 'regular',
        'reg': 'regular'
    })
train_data.to_csv('train_data_cleaned.csv',index=False)
test_data.to_csv('test_data_cleaned.csv',index=False)

test_data=pd.read_csv("test_data_cleaned.csv")
train_data=pd.read_csv("train_data_cleaned.csv")
fat_content_mapping = {
    'low_fat': 0,
    'regular': 1,
}
for data in [train_data,test_data]:
    data['Item_Fat_Content'] = data['Item_Fat_Content'].map(fat_content_mapping)
outlet_size_mapping = {'Small': 1, 'Medium': 2, 'High': 3}
outlet_location_type_mapping = {'Tier 3': 1, 'Tier 2': 2, 'Tier 1': 3}
for df in [train_data, test_data]:
    df['Outlet_Size'] = df['Outlet_Size'].map(outlet_size_mapping)
    df['Outlet_Location_Type'] = df['Outlet_Location_Type'].map(outlet_location_type_mapping)
train_data_encoded = pd.get_dummies(train_data, columns=['Outlet_Type'], prefix='Outlet_Type')
train_data_encoded = train_data_encoded.astype({col: 'int' for col in train_data_encoded.columns if col.startswith('Outlet_Type_')})
test_data_encoded = pd.get_dummies(test_data, columns=['Outlet_Type'], prefix='Outlet_Type')
test_data_encoded = test_data_encoded.astype({col: 'int' for col in test_data_encoded.columns if col.startswith('Outlet_Type_')})
for data in [train_data_encoded,test_data_encoded]:
    data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
def map_item_category(item_identifier):
    if item_identifier.startswith('FD'):
        return 'Food'
    elif item_identifier.startswith('DR'):
        return 'Drink'
    elif item_identifier.startswith('NC'):
        return 'Non_Consumable'
    else:
        return 'Unknown'
train_data_encoded['Item_Category'] = train_data_encoded['Item_Identifier'].apply(map_item_category)
test_data_encoded['Item_Category'] = test_data_encoded['Item_Identifier'].apply(map_item_category)
train_data_encoded = pd.get_dummies(train_data_encoded, columns=['Item_Category'], prefix='Item_Category')
train_data_encoded = train_data_encoded.astype({col: 'int' for col in train_data_encoded.columns if col.startswith('Item_Category_')})
test_data_encoded = pd.get_dummies(test_data_encoded, columns=['Item_Category'], prefix='Item_Category')
test_data_encoded = test_data_encoded.astype({col: 'int' for col in test_data_encoded.columns if col.startswith('Item_Category_')})
columns_to_remove = ['Item_Type', 'Outlet_Establishment_Year']
train_data_final = train_data_encoded.drop(columns=columns_to_remove)
test_data_final = test_data_encoded.drop(columns=columns_to_remove)
train_data_final.to_csv('train_data_final.csv',index=False)
test_data_final.to_csv('test_data_final.csv',index=False)
train_data = pd.read_csv('train_data_final.csv')
test_data = pd.read_csv('test_data_final.csv')
X = train_data.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'], axis=1)
y = train_data['Item_Outlet_Sales']
X_test = test_data.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_val_cnn = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),  
    Dropout(0.3),  
    Conv1D(filters=32, kernel_size=3, activation='relu'),  
    Flatten(),  
    Dense(64, activation='relu'),  
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train_cnn, y_train, epochs=100, batch_size=32, validation_data=(X_val_cnn, y_val), verbose=1)
predictions = model.predict(X_test_cnn).flatten()
predictions = predictions.flatten()
non_negative_predictions = [max(pred, 0) for pred in predictions]
submission = pd.DataFrame({
    'Item_Identifier': test_data['Item_Identifier'],
    'Outlet_Identifier': test_data['Outlet_Identifier'],
    'Item_Outlet_Sales': predictions  
})
submission.to_csv('submission.csv', index=False)

