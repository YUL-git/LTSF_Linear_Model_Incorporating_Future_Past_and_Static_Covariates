import matplotlib.pyplot as plt
import pandas as pd

def plot_sales_and_predictions(past_sales, predicted_sales, start_date="2023-01-01", end_date="2023-04-26"):
    
    # 날짜 범위 생성
    past_dates = pd.date_range(start=start_date, end="2023-04-04")
    future_dates = pd.date_range(start="2023-04-05", end=end_date)
    
    # 그래프 생성
    plt.figure(figsize=(10, 5))
    
    # 과거 판매량
    plt.plot(past_dates, past_sales, color='blue', label='Past Sales')
    
    # 예측 판매량
    plt.plot(future_dates, predicted_sales, color='red', label='Predicted Sales')
    
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Past and Predicted Sales")
    plt.legend()
    plt.grid(True)
    plt.show()

