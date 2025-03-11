rijeci_pjesme = {}
fsong = open("D:\\Downloads\\song.txt")
for line in fsong:
    line = line.strip()
    rijeci = line.split()
    for rijec in rijeci:
        if rijec in rijeci_pjesme:
            rijeci_pjesme[rijec] += 1
        else:
            rijeci_pjesme[rijec] = 1
counter = 0
for key,value in rijeci_pjesme.items():
    if value == 1:
        counter += 1
        print(key)
print (counter)
print (rijeci_pjesme)
fsong.close()
