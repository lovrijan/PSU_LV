file = open('D:\\Downloads\\SMSSpamCollection.txt')

spam_sum = 0
ham_sum = 0
spam_count = 0
ham_count = 0
usklicnik_count = 0


for line in file:
    line = line.strip()
    words = line.split()
    if words[0] == "ham":
        ham_sum += 1
        for i in words:
            ham_count += 1
    elif words[0] == "spam":
        spam_sum += 1
        for i in words:
            spam_count += 1
        word = words[-1]
        if word[-1] == "!":
            spam_count += 1
print("Avg words of ham: ", ham_sum/ham_count)
print("Avg words of spam: ", spam_sum/spam_count)
print("Broj pojavljivanja usklicnika u spam-u: ", spam_count)
file.close()