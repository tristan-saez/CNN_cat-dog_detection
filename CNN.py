"""

Fichier de classe du réseau de neurones.
Dans ce fichier, chaque couche du réseau y est définie.

"""
from torch import nn
import torch


class ImgCnn(nn.Module):

    #
    def __init__(self):
        super().__init__()
        # Première Couche Séquentielle du réseau de neurones (chaque sous-couche sera executée à la suite de l'autre)
        self.conv_layer_1 = nn.Sequential(
            # Couche de convolution - prend la matrice 3D d'une image en entrée et en ressort 64 matrices
            nn.Conv2d(3, 64, 3, padding=1),
            # Ajout d'une couche ReLU permettant de mettre à 0 les valeurs négatives obtenues
            nn.ReLU(),
            # Normalise les données d'un batch (?)
            nn.BatchNorm2d(64),
            # Réduit la taille des matrices de moitié tout en gardant le maximum d'informations
            nn.MaxPool2d(2))
        # Deuxième couche Séquentielle
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(64, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2))
        # Troisième couche Séquentielle
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2))
        # Couche Séquentielle effectuant les étapes de transformation des matrices calculées
        self.classifier = nn.Sequential(
            # Transforme les matrices sous forme de vecteur
            nn.Flatten(),
            # Effectue une application linéaire pour transformer les données vecteurisées en un résultat du type chat : 0 / chien :1
            nn.Linear(in_features=512*3*3, out_features=2))

    # La fonction create permet de mettre les différents layer à la suite.
    def forward(self, x: torch.Tensor):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x