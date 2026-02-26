from Src.Preprocess import load_and_prep_housing
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error , r2_score
import numpy as np
from Src.Models import get_models
import matplotlib.pyplot as plt

def run_experiment():
    X_train,X_test,y_train,y_test,cols = load_and_prep_housing()
    all_models = get_models()
    for name , model in all_models.items():
        model.fit(X_train,y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test,model.predict(X_test))
        print(f"{name} -> R2 score : {score:.4f}")
        plt.figure(figsize=(10,6))
        plt.scatter(y_test,preds,alpha=0.3,color='blue')
        plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--',lw=2)
        plt.title(f"{name} (R2: {score:.4f})")
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        filename = f"{name.replace(' ','_')}_plot.png"
        plt.savefig(filename)
        plt.show()


if __name__ == "__main__":
    print("Script started......")
    run_experiment()
    

    
# # def run_housing_analysis():
# def find_best_depth():
    
#     depths = range(1,21)
#     scores = []
#     for d in depths:
#         model = DecisionTreeRegressor(max_depth=5,random_state=42)
#         # "Linear Regression" : LinearRegression(),       
#         model.fit(X_train,y_train)
#         score = r2_score(y_test,model.predict(X_test))
#         scores.append(score)
#         print(f"Depth {d} : R2 score = {score:.4f}")
#         # pred = model.predict(X_test)
#         # rmse = np.sqrt(mean_squared_error(y_test,pred))
#         # r2 = r2_score(y_test,pred)
#         # print(f"\nModel : {name}")
#         # print(f"RMSE : {rmse:.4f}")
#         # print(f"R2 score : {r2:.4f}")
#     max_idx = scores.index(max(scores))
#     best_depth = list(depths)[max_idx]
#     print(f"\nThe baap depth is {best_depth} with a score of {max(scores):.4f}")
#     print("-"*30)

# if __name__ == "__main__":
#     # run_housing_analysis()
#     find_best_depth()
