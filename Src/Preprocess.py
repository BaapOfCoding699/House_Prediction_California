import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler

def load_and_prep_housing():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data,columns=housing.feature_names) 
    df["MedHouseVal"] = housing.target
    df["Room_per_person"] = df["AveRooms"] / df["AveOccup"]
    df["BedRoom_Ratio"] = df["AveBedrms"] / df["AveOccup"]
    X = df.drop(["AveRooms","AveBedrms","AveOccup","MedHouseVal"],axis = 1)
    y = df["MedHouseVal"]
    X_train , X_test, y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    return X_train_scaled , X_test_scaled , y_train , y_test , X.columns

if __name__ == "__main__":
    X_train , X_test , y_train , Y_test , cols = load_and_prep_housing()
    print("Data loaded Features : {list(cols)}")
    print(f"Train on {X_train.shape[0]} houses.")
