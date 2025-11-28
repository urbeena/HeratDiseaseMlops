# train_models.py
import os
import pickle
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
#from preprocessing import preprocessing_split

#file_path = 'C:\\Users\\HP\\OneDrive\\Desktop\\MLOPS\\HeartDisease\\data\\heart.csv'


#x_train, x_test, y_train, y_test, selected_features, scaler = preprocessing_split(file_path)
def train_models(X_train, y_train):

    models = {
            "XGBoost": {
                "model": XGBClassifier(random_state=42, eval_metric='logloss'),
                "params": {
                    'n_estimators': [100],
                    'max_depth': [3, 5],
                    'learning_rate': [0.05],
                    'subsample': [0.8]
                }
            },
            "RandomForest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    'n_estimators': [100],
                    'max_depth': [5],
                    'min_samples_split': [5]
                }
            },
            "DecisionTree": {
                "model": DecisionTreeClassifier(random_state=42),
                "params": {
                    'max_depth': [3, 5],
                    'min_samples_split': [2, 4]
                }
            }
        }

    for name, cfg in models.items():
            print(f"\n🔍 Training {name}...")
            grid = GridSearchCV(cfg["model"], cfg["params"], cv=3, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            print(f"✅ Best parameters for {name}: {grid.best_params_}")

            with open(f'HeartDisease/models/{name}_model.pkl','wb') as f:
                  pickle.dump(best_model,f)
    print("\n............................")
    print("Best Model: ",best_model)
#train_models(x_train, y_train)