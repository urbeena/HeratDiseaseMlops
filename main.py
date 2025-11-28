from data_loader import load_data
from preprocessing import preprocessing_split
from train import train_models
from evaluate_model import evaluate_model

if __name__ == "__main__":
    file_path = 'C:\\Users\\HP\\OneDrive\\Desktop\\MLOPS\\HeartDisease\\data\\heart.csv'

    #step1: Load data
    df = load_data(file_path)
    print('First 5 rows of the dataset:')
    print(df.head())


    #step2: Preprocess and split data
    X_train, X_test, y_train, y_test, selected_features, scaler=preprocessing_split(file_path)
    # ✅ Step 3: Display Final Information
    print(f"\n✅ Final Split:")
    print(f"🔹 Training Samples: {X_train.shape}")
    print(f"🔹 Testing Samples: {X_test.shape}")
    print(f"📌 Features Used: {selected_features}")


    #step3: Train models
    train_models(X_train, y_train)

    #step4: Evaluate models
    evaluate_model('HeartDisease/models/', X_test, y_test)


