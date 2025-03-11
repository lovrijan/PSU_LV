try:
    ocjena = float(input("Unesi broj između 0.0 - 1.0 "))
    if ocjena <= 0.0 or ocjena >= 1.0:
        print("Brojevi nisu u range-u")
    elif ocjena >= 0.9:
        print("A")
    elif ocjena >= 0.8:
        print("B")
    elif ocjena >= 0.7:
        print("C")
    elif ocjena >= 0.6:
        print("D")
    elif ocjena >= 0.5:
        print("E")
    elif ocjena < 0.5:
        print("F")
    else:
        raise ValueError
except ValueError:
    print("To nije broj")