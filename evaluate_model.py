import os
import pickle
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score,roc_auc_score, precision_score, recall_score
from preprocessing import preprocessing_split
x_train, x_test, y_train, y_test, selected_features, scaler = preprocessing_split('C:\\Users\\HP\\OneDrive\\Desktop\\MLOPS\\HeartDisease\\data\\heart.csv')

model_path='HeartDisease/models/'
def evaluate_model(model_path, X_test, y_test):
    model_files = [f for f in os.listdir(model_path) if f.endswith('_model.pkl')]
    results = {}
    models_dict={}
    for model_file in model_files:
        with open(os.path.join(model_path, model_file), 'rb') as f:
            model = pickle.load(f)

             # Store model for later comparison
            models_dict[model_file] = model

            print(f"\nEvaluating Model: {model_file}")

            
            y_pred= model.predict(X_test)

            accuracy=accuracy_score(y_test,y_pred)
            print(f"Accuracy: {accuracy:.2f}")

            cm=confusion_matrix(y_test,y_pred)
            print("\nConfusion Matrix:")
            print(cm)

            f1=f1_score(y_test,y_pred)
            print(f"F1 Score: {f1:.2f}")

            auc_score = roc_auc_score(y_test, y_pred)
            print(f"ROC AUC Score: {auc_score:.2f}")

            precision_score_value = precision_score(y_test, y_pred)
            print(f"Precision Score: {precision_score_value:.2f}")

            recall_score_value = recall_score(y_test, y_pred)
            print(f"Recall Score: {recall_score_value:.2f}")

            print('..............................')



            results[model_file]={
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc_score': auc_score,
                'precision_score': precision_score_value,
                'recall_score': recall_score_value
            }
            

# 🔍 Select Best Model Based on Accuracy (can change to 'roc_auc' or 'f1_score')
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model= models_dict[best_model_name]
    #save the best_model in models folder
    best_model_path = os.path.join(model_path, "best_model.pkl")
    joblib.dump(best_model, best_model_path)

    return results, best_model_name, best_model


        
evaluate_model(model_path, x_test, y_test)