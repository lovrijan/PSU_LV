import numpy as np
import pandas as pd
# A) zadatak: Kojih 5 automobila ima najveću potrošnju? (koristite funkciju sort)
print("ZADATAK A:")
mtcars=pd.read_csv('C:\\Users\\student\\Downloads\\mtcars.csv')

top5_potrosnja = mtcars.sort_values(by='mpg', ascending=False).head(5)
print("Top 5 automobila s najvećom potrošnjom:")
print(top5_potrosnja[['car', 'mpg']])

# B) zadatak: Koja tri automobila s 8 cilindara imaju najmanju potrošnju?
print("\nZADATAK B:")
min_potrošnja = mtcars[mtcars['cyl'] == 8].sort_values(by='mpg', ascending=False).head(3)
print("Tri automobila s 8 cilindara s najmanjom potrošnjom:")
print(min_potrošnja[['car', 'mpg']])

# C) zadatak: Kolika je srednja potrošnja automobila sa 6 cilindara?
print("\nZADATAK C:")
sred_potr = mtcars[mtcars["cyl"] == 6]["mpg"].mean()
print("Srednja potrošnja automobila sa 6 cilindara iznosi:\n ",round(sred_potr,4),"mpg")

# D) zadatak: Kolika je srednja potrošnja automobila s 4 cilindra mase između 2000 i 2200 lbs?
print("\nZADATAK D:")
sred4 = mtcars[(mtcars.cyl == 4) & (mtcars.wt>=2000) | (mtcars.wt<=2200)]["mpg"].mean()
print("Srednja potrošnja automobila sa 4 cilindara i mase između 2000 i 2200 iznosi:\n ",round(sred4,4),"mpg")

print("\nZADATAK E:")
# E) zadatak: Koliko je automobila s ručnim, a koliko s automatskim mjenjačem u ovom skupu podataka?
counta = (mtcars.am == 1).sum()
countm = (mtcars.am == 0).sum()
print("Automatski:",counta,"\nManualni:",countm)

# F) zadatak: Koliko je automobila s automatskim mjenjačem i snagom preko 100 konjskih snaga?
print("\nZADATAK F:")
kaunt = ((mtcars.am==1) & (mtcars.hp>100)).sum()
print("Broj automobila s automatskim mjenjačem i snagom preko 100 KS iznosi:",kaunt)

# G) zadatak: Kolika je masa svakog automobila u kilogramima?
mtcars['masa_kg'] = mtcars['wt'] * 0.45 * 1000
print(mtcars[['car','masa_kg']])
