import numpy as np
import matplotlib.pyplot as plt

mpg=[]
hp=[]

data = np.loadtxt(open("C:\\Users\\student\\Downloads\\mtcars.csv", "rb"), usecols=(1,2,3,4,5,6),delimiter=",", skiprows=1)
mpg=data[:,0:1]
hp=data[:,3:4]
wt=data[:,5:6]
cyl=data[:,1:2]
plt.scatter(mpg,hp,color='r',s=wt*10)
plt.show()
#print ("CILINDRI:",cyl)
mpgcyl=[]
a=-1
for i in cyl:
    a+=1
    if i==6:
        mpgcyl.append(mpg[a])
        
print("Minimum:",np.min(mpgcyl))
print("Maximum:",np.max(mpgcyl))
print("Avg:",np.mean(mpgcyl))