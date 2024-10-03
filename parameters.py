"""

Classe définissant les paramètres utilisés par le CNN

"""

import os


class Parameters:
    def __init__(self, img_size=(224, 224), n_workers=os.cpu_count(), device='cuda', model_name="cat-n-dog_model.pt"):
        self.img_size = img_size
        self.n_workers = n_workers
        self.device = device
        self.model_name = model_name
        self.class_names = ["chat", "chien"]
