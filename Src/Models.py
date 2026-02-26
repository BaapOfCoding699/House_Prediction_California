from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def get_models():
    return{
        "Linear Regression": LinearRegression(),
        "Decison Tree": DecisionTreeRegressor(max_depth=8,random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300,max_depth=15,min_samples_split=5,random_state=42),
        "XGBoost Power": XGBRegressor(n_estimators = 500,learning_rate=0.05,max_depth=6,random_state=42)
    }
