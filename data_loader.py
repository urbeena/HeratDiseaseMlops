import pandas as pd
file_path = 'C:\\Users\\HP\\OneDrive\\Desktop\\MLOPS\\HeartDisease\\data\\heart.csv'
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully")
        return df
    except Exception as e:
        print(f"Eror in loading data: {e}")
        
