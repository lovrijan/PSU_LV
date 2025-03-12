import numpy as np
import matplotlib.pyplot as plt

x = [1.0,2.0,3.0,3.0,1.0]
y = [1.0,2.0,2.0,1.0,1.0]
plt.axis([0,4.0,0,4.0])
plt.plot(x,y,'b',marker='o', linewidth=2, mfc='m')
plt.xlabel('X os')
plt.ylabel('Y os')
plt.title('Zadatak_1')
plt.show()
