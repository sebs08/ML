### IDEE

Am besten einen Bot erstellen, der durch spielen mit sich selbst eine 'optimale' vier gewinnt Stategie lernt.

D.h. durch den derzeitigen Zustand des Feldes eine optimale Aktion auswählt. 

... das sollte eigentlich ganz gut machbar sein. :)

### Setup

Input: 
* 6 * 7 Spielfeld (Matrix) ~ 42 Input Neuronen
* Netz sieht sich immer als spieler 1
    

Reward : 
* Gewinnen +1; Verlieren -1; sonst 0

Aktionen:
* mögliche Spielzüge gegeben das Feld

Zustände:
* Spielfeld Zustände

Exploration:
* epsilon greedy ? 

Ziel:
* Approximation der Gewinnwahrscheinlichkeit in Abhägigkeit der gewählten Aktion

Netz:
* CNN | DNN | RNN ?
* implementation mittels Tensorflow
* soll Gewichte laden & speichern können
* Architekturvorschlag: 
    - Conv2D Size 2x2 Stride 1
    - Conv2D Size 3x3 Stride 1
    - Dense Layer
    - 1 Dim Output
* Idee: Züge simulieren und Gewinnwahrscheinlichkeit ermitteln (output)
* Netz soll *nicht* lernen was Aktionen bewirken, nur aktionen auswerten (kennt Aktionen)