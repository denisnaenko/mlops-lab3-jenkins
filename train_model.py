import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import mlflow
import joblib


def scale_frame(frame):
    df = frame.copy()
    
    # Преобразуем месяц в числовой
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    df['Month'] = df['Month'].map(month_map)
    
    # Целевая переменная: цена нового авто
    X = df.drop(columns = ['New Price ($)'])
    y = df['New Price ($)']

    # Масштабирование и преобразование
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scaled = scaler.fit_transform(X)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))
    
    return X_scaled, y_scaled, power_trans


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    # Читаем очищенные данные
    df = pd.read_csv("df_clear.csv")
    X, y, power_trans = scale_frame(df)

    # Делим данные
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Гиперпараметры
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        'penalty': ["l1", "l2", "elasticnet"],
        'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'fit_intercept': [False, True],
    }

    # Запускаем эксперимент
    mlflow.set_experiment("linear model cars")
    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train.ravel())
        best = clf.best_estimator_

        # Предсказания
        y_pred = best.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        y_price_val = power_trans.inverse_transform(y_val)

        # Метрики
        rmse, mae, r2 = eval_metrics(y_price_val, y_price_pred)

        # Логируем параметры
        mlflow.log_params({
            "alpha": best.alpha,
            "l1_ratio": best.l1_ratio,
            "penalty": best.penalty,
            "loss": best.loss,
            "fit_intercept": best.fit_intercept,
            "epsilon": best.epsilon
        })
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Логируем модель
        signature = infer_signature(X_train, best.predict(X_train))
        mlflow.sklearn.log_model(best, "model", signature=signature)

        # Сохраняем в файл
        with open("lr_cars.pkl", "wb") as f:
            joblib.dump(best, f)

    # Получаем путь к лучшей модели
    dfruns = mlflow.search_runs()
    path2model = dfruns.sort_values("metrics.r2", ascending=False).iloc[0]['artifact_uri'].replace("file://", "") + "/model"
    print(path2model)
