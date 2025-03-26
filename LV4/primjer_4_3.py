import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ucitavanje ociscenih podataka
df = pd.read_csv('C:\\Users\\student\\Desktop\\LV4_PSU_LB\\PSU_LV\\LV4\\cars_processed.csv')
print(df.info())

# razliciti prikazi
sns.pairplot(df, hue='fuel')

sns.relplot(data=df, x='km_driven', y='selling_price', hue='fuel')
df = df.drop(['name','mileage'], axis=1)

obj_cols = df.select_dtypes(object).columns.values.tolist()
num_cols = df.select_dtypes(np.number).columns.values.tolist()

fig = plt.figure(figsize=[15,8])
for col in range(len(obj_cols)):
    plt.subplot(2,2,col+1)
    sns.countplot(x=obj_cols[col], data=df)

df.boxplot(by ='fuel', column =['selling_price'], grid = False)

df.hist(['selling_price'], grid = False)

#tabcorr = df.corr()
#sns.heatmap(df.corr(), annot=True, linewidths=2, cmap= 'coolwarm')
avg1 = 0.00
avg2 = 0.00
for avg in df:
    if df.fuel == "Dizel":
        avg1=df.fuel.mean()
    elif df.fuel == "Petrol":
        avg2=df.fuel.mean()
print(avg1,avg2)
plt.show()