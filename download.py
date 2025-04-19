import pandas as pd

def download_data():
    df = pd.read_csv('https://raw.githubusercontent.com/denisnaenko/cars_trends_dataset/refs/heads/main/automobile_prices_economics_2019_2023.csv', delimiter=",")
    df.to_csv("cars_trends.csv", index=False)
    return df

def clear_data(path):
    df = pd.read_csv(path)
    
    # Удалим строки с пустыми датами
    df = df.dropna(subset=['Month/Year'])

    # Удалим символы '%' и ',' и преобразуем типы данных
    df['New Price ($)'] = df['New Price ($)'].str.replace(',', '').astype(float)
    df['Used Price ($)'] = df['Used Price ($)'].str.replace(',', '').astype(float)
    df['Inflation Rate (%)'] = df['Inflation Rate (%)'].str.replace('%', '').astype(float)
    df['Interest Rate (%)'] = df['Interest Rate (%)'].str.replace('%', '').astype(float)
    
    df['Units Sold'] = df['Units Sold'].str.replace(',', '')
    df['Units Sold'] = df['Units Sold'].fillna(0).astype(int)

    # Преобразуем дату
    df['Year'] = df['Month/Year'].apply(lambda x: int('20' + x.split('-')[0]))
    df['Month'] = df['Month/Year'].apply(lambda x: x.split('-')[1])
    df.drop(columns=['Month/Year'], inplace=True)

    # Удалим выбросы
    df = df[df['New Price ($)'] > 1000]
    df = df[df['Used Price ($)'] > 500]
    df = df[df['Inflation Rate (%)'] < 50]
    df = df[df['Interest Rate (%)'] < 50]
    df = df[df['Units Sold'] > 0]

    df = df.reset_index(drop=True)
    df.to_csv("df_clear.csv", index=False)

    return True


download_data()
clear_data("cars_trends.csv")


