import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

def generation(points):
    f = open("file.txt", "w")
    for i in range(points):
        x1 = np.random.uniform()
        x2 = np.random.uniform()
        if (x1 + x2 -1 > 0 and x1 > 0.5):
            tag = 1
        else:
            tag = -1
        f.write(str(x1) + " " + str(x2) + " " + str(tag) + "\n")
    f.close()

class Neurone:
    def __init__(self):
        self.pasApp = 0.01
        self.biais = 0.5
        self.sortie = 0
        self.poidsTab = np.random.uniform(0,1,2)

    def neuronalValue(self, x1, x2, tag):
        sigma = self.poidsTab[0]*x1 + self.poidsTab[1]*x2 - self.biais
        if sigma > 0:
            self.sortie = 1
        else:
            self.sortie = -1

    def neuronalUpdate(self, x1, x2, tag):
        self.biais = self.biais + self.pasApp * (tag - self.sortie) * (-0.5)
        self.poidsTab[0] += self.pasApp * (tag - self.sortie) * x1
        self.poidsTab[1] += self.pasApp * (tag - self.sortie) * x2


def main():
    neurone = Neurone()
    erreurTab = []
    for i in range(100):
        nbErr=0
        with open("file.txt", "r") as file:
            for line in file:
                x1, x2, tag = line.split()
                x1 = float(x1)
                x2 = float(x2)
                tag = int(tag)
                neurone.neuronalValue(x1,x2,tag)
                if(neurone.sortie != tag):
                    nbErr += 1
                    neurone.neuronalUpdate(x1,x2,tag)
        erreurTab.append(nbErr)
        print(nbErr)
   # plt.plot(erreurTab)
  #  plt.show() 

if __name__ == '__main__':
	main()

generation(100)

'''
Plus le pas d'apprentissage est grand, plus il y a d'erreurs. (Je ne suis pas sur de moi)

Plus le nombre d'exemple sera grands, plus il sera difficile de pouvoir séparer nos données avec une droite.
Par exemple, pour 100 tuples de points, il est quasi-impossible, avec les valeurs de l'exercice 3, de pouvoir séparer notre jeu de données. 
Alors que pour 10 tuples, c'est quasiment toujours possible.
'''