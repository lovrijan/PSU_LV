flocation = "D:\\Downloads\\"
fname = input("Ime datoteke: ")
flocation += fname
file = open(flocation)
average = 0.0
count = 0

for line in file:
    line = line.strip()
    if line.startswith("X-DSPAM-Confidence:"):
        count += 1
        lsplit = line.split(": ")
        num = lsplit[1]
        average += float(num)
average /= float(count)
print("Average X-DSPAM-Confidence: ", average)
file.close()