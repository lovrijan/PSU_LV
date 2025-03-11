def total_euro():
    rezultat = sati * satnica
    print("Ukupno:", float(rezultat), "eura")

sati = int(input("Unesi broj sati: "))
satnica = float(input("Unesi satnicu: "))

print("")
print("Radni sati:", sati, "h")
print("eura/h:", satnica)
total_euro()