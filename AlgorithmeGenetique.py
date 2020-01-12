#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:03:28 2019

@author: camillemac
"""

import numpy as np
import random

class Individu:
    
    def __init__(self, _parameters, _captures =[], _fitness= 100000000, _proba = 0, _plage = [] ):
        self._parameters = _parameters
        self._captures = _captures
        self._fitness = _fitness
        self._proba = _proba 
        self._plage = _plage
    
    #getters and setters  
    @property
    def parameters(self):
        return self._parameters
    
    @parameters.setter 
    def parameters(self,parameters):
        self._parameters = parameters
        
    @property
    def captures(self):
        return self._captures
    
    @captures.setter 
    def captures(self,captures):
        self._captures = captures
        
    @property
    def fitness(self):
        return self._fitness
    
    @fitness.setter 
    def fitness(self,fitness):
        self._fitness = fitness
        
    @property
    def proba(self):
        return self._proba
    
    @proba.setter 
    def proba(self,proba):
        self._proba = proba
        
    @property
    def plage(self):
        return self._plage
    
    @plage.setter 
    def plage(self,plage):
        self._plage = plage
        
    def simulationCapture(self):
        effortValues = [0.0,0.2,0.1,0.7,0.5,0.0,1.0,0.3,0.9,0.9,0.9,0.0,0.3,0.5,1.0,0.9,0.8,1.2,0.0,0.7,0.8]
        mCaptures = []
        
        r = self.parameters["r"]
        q = self.parameters["q"]
        k = self.parameters["k"]
        #La biomasse totale ne peut pas être supérieure à la capacité du milieu
        if self.parameters["b0"] > k:
            self.parameters["b0"] = k
        biomasse = self.parameters["b0"]
        for e in effortValues:
            capture = q * e * biomasse
            biomasse = biomasse +  r * (1 - biomasse/k) * biomasse - q * e * biomasse
            if biomasse <0:
                biomasse = 0
            if biomasse > k:
                biomasse = k
            mCaptures.append(capture)
        self.captures = mCaptures

    def calculFitness(self, capturesVeritables):
        capturesSimulees = self.captures
        self.fitness = 0
        for i in range(len(capturesSimulees)):
            self.fitness = self.fitness + np.sqrt(np.power(capturesVeritables[i] - capturesSimulees[i], 2))
            
   
    def returnIndividu(self):
        individu = {
            "r" : self.parameters["r"],
            "q" : self.parameters["q"],
            "k" : self.parameters["k"],
            "b0" : self.parameters["b0"],
            "captures" : self.captures,
            "fitness" : self.fitness,
            "probabilite" : self.proba}
        return individu


class Generation:
    
   def __init__(self, _individus, _mutation_rate = 0):
        self._individus = _individus
        self._mutation_rate = _mutation_rate

   @property
   def individus(self):
       return self._individus
    
   @individus.setter 
   def individus(self,individus):
       self._individus = individus
        
#Calcul de la probabilité pour chaque individu d'etre sélectionné dans la génération en cours
   def calculProba(self):
       individus = self.individus
       sommeFitness = 0
       for i in individus:
           sommeFitness = sommeFitness + 1/i.fitness
       for i in individus:
           i.proba= (1/i.fitness)/sommeFitness
       return individus
   
   #Définition du segment de probabilité, défintion d'une plage pour chaque individu en fonction de la génération courante
   def calculPlage(self):
       self.individus = np.random.permutation(self.individus)
       self.calculProba()
       for i in range(len(self.individus)):
           if i==0:
               self.individus[i].plage = [0,self.individus[i].proba]
           else:
               self.individus[i].plage = [self.individus[i-1].plage[1], self.individus[i-1].plage[1]+self.individus[i].proba]
           #print( self.individus[i].plage )
   
   #Sélection du nombre d'individu demandé 
   def selection(self, nbIndividu):
       self.calculPlage()
       selectedIndividus = []
       for i in range(nbIndividu+1):
           selector = i/(nbIndividu+1)
           for p in self.individus:
               if selector> p.plage[0] and selector<=p.plage[1]:
                   selectedIndividus.append(p)
                   break
       #print(len(selectedIndividus))
       return selectedIndividus
          
   
   def mutation(self):
       #On ré indexe notre tableau d'individu aléatoirement
       self.individus = np.random.permutation(self.individus)
       rValues = [e.parameters["r"] for e in self.individus[0::2]]
       qValues  = [e.parameters["q"] for e in self.individus[0::2]]
       kValues = [e.parameters["k"] for e in self.individus[0::2]]
       b0Values = [e.parameters["b0"] for e in self.individus[0::2]]
       variables = [1,2,3,4]

       #On effectue la mutation sur 50% de la génération donnée, 2 pour en sélectionner 1/2 = 50%
       for i in range(0,len(self.individus),2):
           #Nombre de variables à modifier
           selector = random.randint(1,4)
           #numéro des variable à modifier permutés aléatoirement, 1 pour r, 2 pour q, 3 pour k, 4 pour b0
           variables = np.random.permutation(variables)
           #selector = random.randint(1,4) #On génère le nombre de parametres à changer
           while selector > 0:
               if variables[selector-1] == 1:
                   mu = self.individus[i].parameters["r"]
                   msigma = self.sigma(rValues)
                   while True:
                       mR = np.random.normal(mu, msigma)
                       if mR <= 0.5 and mR>=0:
                           self.individus[i].parameters["r"] = mR   
                           break
               elif variables[selector-1] == 2:
                   mu = self.individus[i].parameters["q"]
                   msigma = self.sigma(qValues)
                   while True:
                       mQ = np.random.normal(mu, msigma)
                       if mQ <= 0.5 and mQ>=0:
                           self.individus[i].parameters["q"] = mQ
                           break
               elif variables[selector-1] == 3:
                   mu = self.individus[i].parameters["k"]
                   msigma = self.sigma(kValues)
                   while True:
                       mK = np.random.normal(mu, msigma)
                       if mK <= 2000 and mK>=100:
                           self.individus[i].parameters["k"] = mK
                           break
               elif variables[selector-1] == 4:
                   mu = self.individus[i].parameters["b0"]
                   msigma = self.sigma(b0Values)
                   while True:
                       mB0 = np.random.normal(mu, msigma)
                       if mB0 <= 2000 and mB0>=100:
                           self.individus[i].parameters["b0"] = mB0
                           break
               selector = selector - 1
           self.individus[i].simulationCapture()
           
   def sigma(self, vValues):
       return max(vValues) - np.sum(vValues)/len(vValues)
            
            
"""

-------MAIN-------

"""        
   
#Définition de la loi normale initialisée aux valeurs mu = 0.5 et sigma = 0.2
def loiNormale(mu = 0.5, sigma = 0.2):
    #Equivalent du do while w > 1 en python
    while True: 
        alea1 = random.uniform(0, 1)
        alea2 = random.uniform(0, 1)
        w = np.power(alea1,2) + np.power(alea2,2)
        if w <= 1 and w >= 0:
            return (mu + alea1 * sigma * np.sqrt((-2*np.log(w))/w))
        
#On crée un tableau de couple de parents
#Afin d'éviter des individus présents dans plusieurs couples on sépare en deux parties distinctes parent1 et parent2
def creationCouple(parents):
    couples = []
    parents1 = parents[0::2] #Les parents 1 sont les individus aux rangs pairs dans la selection
    parents2 = parents[1::2] #Les parents 2 sont les individus aux rangs impairs dans la selection
    for i in range(min(len(parents1),len(parents2))):
        couples.append([parents1[i], parents2[i]])
    return couples

#Fonction permettant de générer/retourner deux enfants à partir d'un couple
def croisement(parents):
    individuVeritableParameters = {"r": 0.278, "q": 0.222, "k": 1055, "b0" : 800}
    individuVeritable = Individu(individuVeritableParameters)
    individuVeritable.simulationCapture()
    parent1 = parents[0]
    parent2 = parents[1]
    alpha1 = loiNormale()
    alpha2 = loiNormale()
    alpha3 = loiNormale()
    alpha4 = loiNormale()
    
    enfant1 = Individu({"r": alpha1*parent1.parameters["r"]+((1-alpha1)*parent2.parameters["r"]),
                        "q": alpha2*parent1.parameters["q"]+((1-alpha2)*parent2.parameters["q"]),
                        "k": alpha3*parent1.parameters["k"]+((1-alpha3)*parent2.parameters["k"]),
                        "b0": alpha4*parent1.parameters["b0"]+((1-alpha4)*parent2.parameters["b0"])})
    enfant2 = Individu({"r": alpha1*parent2.parameters["r"]+((1-alpha1)*parent1.parameters["r"]),
                        "q": alpha2*parent2.parameters["q"]+((1-alpha2)*parent1.parameters["q"]),
                        "k": alpha3*parent2.parameters["k"]+((1-alpha3)*parent1.parameters["k"]),
                        "b0": alpha4*parent2.parameters["b0"]+((1-alpha4)*parent1.parameters["b0"])})
    enfant1.simulationCapture()
    enfant2.simulationCapture()
    enfant1.calculFitness(individuVeritable.captures)
    enfant2.calculFitness(individuVeritable.captures)
    enfants = [enfant1, enfant2]
    return enfants


#On sélectionne les "nbIndividus" meilleurs individus de la génération données
def selection(nbIndividus, generation):
    listeProba = [p.proba for p in generation.individus]
    percentageToTake = 1 - nbIndividus/len(generation.individus) 
    #Parmis l'ensemble des probabilité de la génération on obtient la valeur à partir de laquelle on aura les meilleurs individus
    mediane = np.quantile(listeProba, percentageToTake)
    selection = []
    for p in generation.individus:
        if p.proba >= mediane and len(selection)<=nbIndividus:
            selection.append(p)
    return selection

            
def main():
    
    #création de l'individu véritable
    individuVeritableParameters = {"r": 0.278, "q": 0.222, "k": 1055, "b0" : 800}
    individuVeritable = Individu(individuVeritableParameters)
    individuVeritable.simulationCapture()
    print(individuVeritable.returnIndividu())
    
    #initialisation de la premiere génération
    premiereGeneration = []
    nbIndividus = 100
    for i in range(nbIndividus):
        mR = random.uniform(0.05, 0.5)
        mQ = random.uniform(0.05, 0.5)
        mK = random.uniform(100, 2000)
        mB0 = random.uniform(100, 2000)
        individuParameters = {"r": mR, "q": mQ, "k": mK, "b0" : mB0}
        mIndividu = Individu(individuParameters)
        mIndividu.simulationCapture()
        mIndividu.calculFitness(individuVeritable.captures)
        premiereGeneration.append(mIndividu)

    
    #Création première génération
    premiereGeneration = Generation(premiereGeneration)
    
    
    #On initialise un fitness très grand que l'on va tenter de réduire à chaque itération 
    meilleurFitness = 100000000000
    
    #On relève le nombre d'itération
    nbIteration = 0
    
    #Début de la méthode globale. Tant que l'on a pas un fitness satisfaisant on re-itère
    while True:
        
        nbIteration = nbIteration + 1
        
        #On met à jour le meilleur fitness dans la nouvelle generation
        fitnessInGeneration = [i.fitness for i in premiereGeneration.individus]
        meilleurFitness = min(fitnessInGeneration)
        print("meilleur fitness")
        print(meilleurFitness)
        
        #Croisement
        #Calcul de la proba pour chaque individu
        premiereGeneration.calculProba()
        
        #Selection de 40 individus
        selectionParents = []
        #selectionParents = premiereGeneration.selection(40)
        selectionParents = selection(40, premiereGeneration)
        
        #Creation des couples
        mesCouples = creationCouple(selectionParents)
        #Generation des enfants
        mesEnfants = []
        for p in mesCouples:
            coupleEnfant = croisement(p)
            for e in coupleEnfant:
                mesEnfants.append(e)
        generationEnfant = Generation(mesEnfants)
        generationEnfant.calculProba()
        
        #Mutation generation precedente
        #Note: ici le tableau est re indexé donc la plage des individus n'est plus la meme après l'opération
        premiereGeneration.mutation()
        premiereGeneration.calculProba()
        premiereGeneration.calculPlage()
        
        #On recalcul le fitness puis la proba
        for p in premiereGeneration.individus:
            p.calculFitness(individuVeritable.captures)
        premiereGeneration.calculProba()
        
        #On a notre population totale:
        populationTotale = [*generationEnfant.individus, *premiereGeneration.individus]
        populationTotaleGeneration = Generation(populationTotale)
        populationTotaleGeneration.calculProba()
        #Selection: On revient à 100 individus
        #nouvelleGeneration = populationTotaleGeneration.selection(100)
        nouvelleGeneration = selection(100, populationTotaleGeneration)
        premiereGeneration = Generation(nouvelleGeneration)

        
        """
        for p in premiereGeneration.individus:
            print(p.returnIndividu())
            print(nbIteration)"""
            
        #limite à 100 itérations
        if nbIteration > 1000 or meilleurFitness <= 1:
            break

    print("Premier meilleur fitness à la génération " + str(nbIteration))
    
    return

    
if __name__ == "__main__":
    main()
    
    
    
    #créer un objet génération?