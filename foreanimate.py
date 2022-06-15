import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()

fig.figsize = (8,4)

ax1 = fig.add_subplot(1,1,1)

def animate(i):

    df = pd.read_csv('realtime_demand.csv')
    actual_values = df.iloc[:,0].values
    
    forecast_values = df.iloc[:,1].values
    

    if len(actual_values)>=120:
        actual_values = df.iloc[-120: 0].values
        forecast_values =df.iloc[-120: 1].values

    xs = list(range(1, len(actual_values)+1))
    print(xs)
    ax1.clear()
    ax1.plot(xs, actual_values)
    ax1.plot(xs, forecast_values)

    ax1.set_title('One Hour Ahead Load Forecast in MW', fontsize = 32)
    ax1.legend(['Actual','Forecast'], loc = 'lower right')

ani = animation.FuncAnimation(fig, animate, interval =500)

plt.tight_layout()
plt.show()






