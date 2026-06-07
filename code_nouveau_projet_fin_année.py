import numpy as np
from scipy.integrate import quad
from scipy.signal import find_peaks
import sys
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def g(x): #fonction à utiliser
    return x**3-x**2

def verifier_g(): #on ne peut pas vérifier que g est C^2 donc on suppose qu"elle l'est 
     # verifier que g(0) = 0
    if abs(g(0)) > 1e-9: # marge d'erreur pour les erreurs d'arrondi de l'ordi
        print( " g(0) n'est pas égal à 0")
        return False
    else: 
        print( "g(0) = 0 ")
    h = 1e-5
    g_prime = (g(h) - g(-h)) / (2 * h) # Dérivée par différences finies centrées
    if abs(g_prime) > 1e-5:
        print ("g'(0) n'est pas égale à 0 ")
        return False
    else :
        print("g'(0) = 0 ")
        return True

if verifier_g() == False:
    sys.exit() #on arrête le programme si ça vérifie pas les conditions 
    

def G(x):
    # quad = calcule de l'intégrale (formule de quadrature, estimation)
    #le [0] c'est l'estimation de l'erreur, car quad renvoie 2 infos le résultat et l'estimation de l'erreur (de combien il s'est trompé)
    return quad(g, 0, x)[0]

def P(x) : 
    if x == 0 : #on ne peut pas diviser par 0 dans la formule de P 
        return 0
    else : 
        return -2 * G(x) / (x**2)

def parametres(x_max=10.0, nb_points=1000):
    # On commence à 0.001 pour ignorer la solution triviale (c=0)
    x_abscisse = np.linspace(0.001, x_max, nb_points)
    P_val = np.zeros(nb_points)
    
    for i, x in enumerate(x_abscisse):
        P_val[i] = P(x) #pour chaque i = 0,1,...jusqu'a 1000 on calcule la valeur de P au point x
        
    sommets, _ = find_peaks(P_val)
    for i in sommets: 
        c = x_abscisse[i] 
        mu = P_val[i] # u = P(c), P(c) qu'on a calculé pour chaque valeur de i 
        if mu > 0: #u doit etre > 0 
            # Slicing : on vérifie que toute la courbe AVANT le sommet est plus basse
            if np.all(P_val[:i] < mu ): 
                print(f" c = {c}  , mu = {mu}")
                return c, mu 
            
    return None, None
    
c, mu  = parametres()
if c is None or mu is None:
    print("\nErreur : Aucun paramètre valide (c, mu) n'a été trouvé.")
    print(" La fonction P(x) n'a pas de sommet.")
    sys.exit()
    
    
def graphique(g, c, mu, x_max = 10):
    # Résolution de l'équation différentielle pour tracer Q(x)
    Q0 = c / 2
    dQ0 = -np.sqrt(mu * (Q0**2) + 2 * G(Q0))
    
    def equation(x, Y):
        # On transforme notre équa diff d'ordre 2 en un système d'ordre 1
        Q = Y[0]
        dQ = Y[1]
        d2Q = mu * Q + g(Q)
        return [dQ, d2Q]
    # On résout en partant du centre, vers la droite puis la gauche
    sol_droite = solve_ivp(equation, [0, x_max], [Q0, dQ0], max_step=0.1)
    sol_gauche = solve_ivp(equation, [0, -x_max], [Q0, dQ0], max_step=0.1)
    # On recolle les deux morceaux inversés pour avoir l'axe x dans le bon sens
    x_total = np.concatenate((sol_gauche.t[::-1], sol_droite.t))
    Q_total = np.concatenate((sol_gauche.y[0][::-1], sol_droite.y[0]))
    
    plt.figure(figsize=(10, 6))

 
    plt.plot(x_total, Q_total, label="Solution numérique Q(x)", color="blue", linewidth=2.5)

    # Les asymptotes théoriques 
    plt.axhline(y=c, color='red', linestyle='--', label=f"Asymptote en -∞ (c = {c:.2f})")
    plt.axhline(y=0, color='green', linestyle='--', label="Asymptote en +∞ (0)")


    plt.title("Graphe de la fonction Q", fontsize=14, fontweight='bold')
    plt.xlabel("x", fontsize=12)
    plt.ylabel("Q(x)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.show()
    
graphique(g, c, mu)