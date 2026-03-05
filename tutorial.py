import torch
from torch_draw import *
device = DEFAULT_DEVICE

canvas = Canvas(800, 800) # tworzymy płótno
col = color().col # helper 

N_lines = 100
N_points = 100_000
canvas.setTitle("Animated Testing - press esc to exit")
# Rysowanie punktów i linii w pętli - animacja
while canvas.display(1) == False:
    canvas.clear() # czyścimy płótno na czarno
    canvas.clear(color().light_gray) # można explicite wyczyścić na dany kolor
    
    # jeśli używacie GPU to trzeba przenieść tensory na GPU używając, device=DEFAULT_DEVICE.
    # tworzymy N_lines linii: tensor z liczbami od 100 do 700 o wymiarach (N,4) - N linii a 4 bo każda linia ma dwa punkty: początekowy (x,y) i końcowy (x,y)
    lines = torch.randint(100, 700, (N_lines, 4), device=device)
    # tworzymy N_points punktów: tensor z liczbami od 0 do 800 o wymiarach(N,2) - N punktów, każdy ma dwie współrzędne: x i y
    points = torch.randint(0, 800, (N_points, 2), device=device)
    
    # deklarujemy kolory (R,G,B) dla punktów i linii
    colorsLines = col(255, 0, 0) # linie czerwone
    colorsPoints = color().cyan # punkty na cyjanowy (0,255,255)
    
    # rysujemy punkty
    canvas.draw(points,colorsPoints)
    # rysujemy linie
    canvas.drawLine(lines,colorsLines)
    
# Rysowanie obrazów z tablicy 2d
while canvas.display(1) == False:
    # Jeśli mamy tablice 2d liczb to możemy ją przekazać do funkcji draw: W,H,3 - 3 kolory
    tablica = torch.randint(0, 255, (800,400,3), device=device) 
    canvas.draw(tablica)
    
    # Lub W,H - wtedy będzie to obraz w odcieniach szarości
    tablica = torch.randint(0, 255, (400,800), device=device) 
    canvas.draw(tablica)