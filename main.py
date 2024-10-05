import os
import torch
from timeit import default_timer as timer
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from CNN import ImgCnn
from torch import nn

def main():
    # Paramètres d'image
    IMAGE_SIZE = (224, 224)

    # Paramètres d'executions
    # Nombre de sous-tâches utilisées. La valeur est ici définie par le nombre de coeur du CPU
    NUM_WORKERS = os.cpu_count()

    # Choisi d'utiliser CUDA permettant d'effectuer les calculs d'entraînement à l'aide du GPU plutôt que du CPU
    device = 'cuda'
    # Attribue le nom des dossiers où se trouvent les données
    train_dir = "data/training_set"
    test_dir = "data/test_set"

    # Défini la graine sur laquelle la génération de valeurs aléatoires va se baser.
    # obligatoire au bon fonctionnement de pytorch
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Nombre d'epoch. Nombre de fois que l'algorithme sera entraîné
    NUM_EPOCHS = 10

    # Modifications appliqués à l'image avant utilisation.
    data_transform = transforms.Compose([
        # Redimensionne les images de façon à être utilisable par l'algorithme
        transforms.Resize(size=IMAGE_SIZE),
        # Permet de paralléliser les matrices laissant la possibilité au GPU d'effectuer les calculs
        transforms.ToTensor()
    ])

    # Creation des datasets d'apprentissage et de test
    train_data = datasets.ImageFolder(root=train_dir, transform=data_transform, target_transform=None)
    test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

    print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
    print(f"Classes possibles : {train_data.classes}")

    # Analyse des features dans l'image
    img, label = train_data[0][0], train_data[0][1]
    print(f"Image tensor:\n{img}")
    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")

    # Création des Dataloader (s'occupent de charger les données et les répartir dans différents batch)
    # Dataloader d'entraînement
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=16,
                                  num_workers=NUM_WORKERS,
                                  shuffle=True)

    # Dataloader de test
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=1,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False)

    model = ImgCnn().to(device)

    # =================== test du modèle sur une image unique ===================
    # # Récupère un batch d'images
    # img_batch, label_batch = next(iter(train_dataloader))
    # # Extrait une image du batch
    # img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    #
    # # Passe l'image à travers le réseau de neurones et récupère la prédiction
    # model.eval()
    # with torch.inference_mode():
    #     pred = model(img_single.to(device))
    #
    # #
    # print(f"Probabilité de chaque label :\n{torch.softmax(pred, dim=1)}\n")
    # print(f"Label prédit :\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    # print(f"Label réel : {label_single}")
    # =================== test du modèle sur une image unique ===================

    # Défini la fonction de perte. L'objectif est d'obtenir le moins de perte possible entre les entraînements.
    # Cela voudra dire que l'alogrithme n'est pas aléatoire dans ses réponses.
    loss_fn = nn.CrossEntropyLoss()
    # Défini l'optimisateur. La fonction ou algortihme s'occupe d'ajuster les paramètres telles que les poids ou la vitesse d'entraînement
    # Il influe donc sur la précision et la perte.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    # Lance l'entraînement du modèle compilé
    model_results = train(model=model,
                          train_dataloader=train_dataloader,
                          test_dataloader=test_dataloader,
                          optimizer=optimizer,
                          loss_fn=loss_fn,
                          epochs=NUM_EPOCHS,
                          device= device)


def train_cnn(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device):

    # Défini le modèle en mode apprentissage
    model.train()

    # Initialise les valeurs de perte dans l'apprentissage et la précision
    train_loss, train_acc = 0, 0

    # Pour chaque set de données d'entraînement, l'entraînement a lieu
    for batch, (x_train, y_train) in enumerate(dataloader):
        # Envoie les données d'entraînement à CUDA
        x_train, y_train = x_train.to(device), y_train.to(device)

        # Lance la prédiction du modèle (forward)
        y_pred = model(x_train)

        # Recalcule la perte à l'aide de la fonction définie précedemment
        loss = loss_fn(y_pred, y_train)
        train_loss += loss.item()

        # Réinitialise les gradients de l'optimisateur.
        # Cela est nécessaire pour être sûr qu'il ne s'accumuleront pas lors du passage retour dans le CNN
        # L'optimisateur pourrait avoir des résultats faussés sans cela
        optimizer.zero_grad()

        # Calcule la dérivé de la perte par chaque paramètre. Cela permet d'obtenir les gradients
        loss.backward()

        # Modifie les valeurs des paramètres grâce aux valeurs des gradients calculés précédemment
        optimizer.step()

        # Calcule les labels prédits
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        # Les compare avec les labels réels et calcule la précision
        train_acc += (y_pred_class == y_train).sum().item() / len(y_pred)

    # Ajuste les métriques en fonction du nombre de batchs
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_cnn(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, device):
    # Défini le modèle en mode apprentissage
    model.eval()

    # Initialise les valeurs de perte dans l'apprentissage et la précision
    test_loss, test_acc = 0, 0

    # Met le Pytorch en mode inférence désactivant des outils de tracking utile lors de l'apprentissage
    # Cela permet d'obtenir de meilleures performances
    with torch.inference_mode():
        # Pour chaque set de données de test, l'entraînement a lieu
        for batch, (x_test, y_test) in enumerate(dataloader):
            # Envoie les données d'entraînement à CUDA
            x_test, y_test = x_test.to(device), y_test.to(device)

            # Lance la prédiction du modèle
            test_pred_logits = model(x_test)

            # Recalcule la perte à l'aide de la fonction définie précedemment
            loss = loss_fn(test_pred_logits, y_test)
            test_loss += loss.item()

            # Calcule les labels prédits
            test_pred_labels = test_pred_logits.argmax(dim=1)
            # Les compare avec les labels réels et calcule la précision
            test_acc += ((test_pred_labels == y_test).sum().item() / len(test_pred_labels))

    # Ajuste les métriques en fonction du nombre de batchs
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module, device,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,):
    """

    Fonction globale d'entraînement. Elle effectue à la fois l'entraînement puis le test.
    """

    # Met les résultat dans un dictionnaire
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # 3. Pour chaque époch effectue l'entraînement et le test de l'algorithme
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_cnn(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        test_loss, test_acc = test_cnn(model=model, dataloader=test_dataloader, loss_fn=loss_fn,  device=device)

        # Affichage des informations d'entraînement
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Ajout au dictionnaire de résultats
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Enregistrement du model
    torch.save(model, 'trained_model/cat-n-dog_model.pt')

    # Renvoie les résultats à la main
    return results


if __name__ == '__main__':
    main()