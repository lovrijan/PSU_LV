counter = 0
min=0
max = 0
lista = []
brojevi=0

while True:
    x=str(input("Unesi broj "))
    if (str(x) == "Done"):
        break
    counter += 1
    lista.append(x)
    if(int(x) > int(max)):
        max = x
print("Petlja je završena")
print("Korisnik je unio",counter,"brojeva")
lista.sort
for i in range(len(lista)):
    brojevi+=int(lista[i])
    min=lista[0]
    if(int(lista[i])<int(min)):
        min = lista[i]
srednja = brojevi/counter
print(lista)
print("Min:",min)
print("Max:",max)
print("Srednja vrijednost:",srednja)