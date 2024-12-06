# filename: get_tesla_stock_data.py
import yfinance as yf
import matplotlib.pyplot as plt
import datetime

today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)
tesla = yf.Ticker("TSLA")
hist = tesla.history(start=yesterday, end=today, interval="1m")

plt.figure(figsize=(12,6))
plt.plot(hist['Close'])
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Tesla Stock Price Change - Last Day')
plt.grid(True)
plt.show()
