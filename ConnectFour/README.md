### IDEE

Am besten einen Bot erstellen, der durch spielen mit sich selbst eine 'optimale' vier gewinnt Stategie lernt.

D.h. durch den derzeitigen Zustand des Feldes eine optimale Aktion auswählt. 

... das sollte eigentlich ganz gut machbar sein. :)

### Setup

Input: 
* 6 * 7 Spielfeld (Matrix) ~ 42 Input Neuronen
* Netz sieht sich immer als spieler 1
    

Reward : 
* Gewinnen +1; Verlieren -1

Aktionen:
* mögliche Spielzüge gegeben das Feld

Zustände:
* Spielfeld

Exploration:
* epsilon greedy ? 

Ziel:
* Approximation der Gewinnwahrscheinlichkeit in Abhägigkeit der gewählten Aktion

Netz:
* CNN | DNN | RNN ?

