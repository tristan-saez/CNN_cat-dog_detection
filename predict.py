import torchvision
import torch
from torchvision import datasets, transforms
from parameters import Parameters


def predict(img):
    # Couleurs de la console
    prediction = '\033[93m'
    entree = '\033[94m'
    reset = '\033[0m'

    # Récupère les paramètres globaux
    params = Parameters()

    # Récupère l'image avec pytorch
    custom_image = torchvision.io.read_image(img).type(torch.float32)

    # Divise les pixels de l'image par 255 pour obtenir une valeur entre 0 et 1
    custom_image = custom_image / 255.

    # Modifie la taille de l'image
    custom_image_transform = transforms.Compose([
        transforms.Resize(params.img_size),
    ])
    custom_image_transformed = custom_image_transform(custom_image)

    # Charge le modèle
    model = torch.load("trained_model/"+params.model_name, weights_only=False)

    # Pose le modèle en mode évaluation (comme pour le test)
    model.eval()

    # Met le Pytorch en mode inférence désactivant des outils de tracking utile lors de l'apprentissage
    # Cela permet d'obtenir de meilleures performances
    with torch.inference_mode():
        # Envoie l'image pour la prédiction en ajoutant une dimension
        custom_image_pred = model(custom_image_transformed.unsqueeze(dim=0).to(params.device))
        # Transforme les valeurs obtenues en pourcentage de probabilité via softmax
        custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
        # Transforme les pourcentages de probabilité en label binaire
        custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)

        # Convertit les labels en valeurs litérales (chat / chien)
        custom_image_pred_class = params.class_names[custom_image_pred_label.cpu()]
        print(f"{reset}L'image est : {prediction}{custom_image_pred_class}{reset}")


if __name__ == "__main__":
    print("\n["+"="*20+"PREDICTION CHAT/CHIEN"+"="*20+"]")
    print("Entrez le nom de l'image. (avec l'extension d'image)\n")
    img = "img_to_predict/"+str(input(">\033[38;5;206m"))
    predict(img)
