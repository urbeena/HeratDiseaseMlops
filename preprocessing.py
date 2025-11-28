import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

def preprocessing_split(path):
    df = pd.read_csv(path)
    print('Shape(rows,columns):', df.shape)

    # __________ Handle Missing Values __________
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        df.dropna(inplace=True)
        print(f" Dropped rows with missing values: {missing_values}")
    else:
        print("No missing values found ✔")

    # __________ Remove Duplicates __________
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df.drop_duplicates(inplace=True)
        print(f"🗑 Dropped duplicate rows: {duplicate_count}")
    else:
        print("No duplicates found ✔")

    # __________ Separate columns __________
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if "HeartDisease" in numeric_cols:
        numeric_cols.remove("HeartDisease")  # target should not be scaled

    # __________ Encode Categorical Columns FIRST __________
    encoders = {}
    if categorical_cols:
        for col in categorical_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder
        print(f" Encoded categorical columns: {categorical_cols}")
    else:
        print("No categorical columns found ✔")

    # Save encoders
    with open("HeartDisease/models/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    # __________ Split Features / Target __________
    X = df.drop("HeartDisease", axis=1)
    Y = df["HeartDisease"]
    print("\nTarget distribution:")
    print(Y.value_counts())

    # __________ Scale ONLY numeric columns __________
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Save scaler
    with open('HeartDisease/models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # __________ Feature Selection __________
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_scaled, Y)

    selector = SelectFromModel(rf, threshold='median', prefit=True)
    X_selected = selector.transform(X_scaled)

    selected_features = np.array(X_scaled.columns)[selector.get_support()].tolist()
    print(f"\n🎯 Selected Features: {selected_features}")

    # Save selected features list
    with open('HeartDisease/models/selected_features.txt', 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")

    # __________ Train-test split __________
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, Y, test_size=0.2, stratify=Y, random_state=42
    )

    return X_train, X_test, y_train, y_test, selected_features, scaler



#(preprocessing_split('C:\\Users\\HP\\OneDrive\\Desktop\\MLOPS\\HeartDisease\\data\\heart.csv'))



'''
"If we do not save the scaler, selector, encoder, etc., then every time we predict something, 
the model may be retrained and the preprocessing values (mean, standard deviation, feature importance,
 encodings) will change — which will make predictions inconsistent and unreliable.
 #
 Best Principle

Inference must use EXACT same preprocessing used during training.

Not similar.
Not re-calculated.
Not assumed.
Not approximated.

But exact same objects (scaler, encoder, selector, feature order)."'''