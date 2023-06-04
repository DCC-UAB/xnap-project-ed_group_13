import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from AFAD.afad_coral import resnet34
import easygui






'''
# Per carregar un model primer has de crear l'arquitectura en questió i despres carregues el fitxer .pt de torch amb un load
'''

DATASET = 'cacd' if int(input('seleccionar el dataset del model CACD(1)/AFAD(2)\n Escull:')) == 1 else 'afad'

MODEL = 'inicial' if int(input('seleccionar quin model es vol inicial(1)/final(2)\n Escull:')) == 1 else 'final'


if DATASET == 'cacd':
    ADD_CLASS = 14 # add_class afageix un offset al output ja que els 
                   # labels amb els que treballen els models començen 
                   # a 0 i no a l'edat minima dels datasets
    NUM_CLASSES = 49
elif DATASET == 'afad':
    ADD_CLASS = 15 
    NUM_CLASSES = 26


GRAYSCALE = False


model34 = resnet34(NUM_CLASSES, GRAYSCALE) # creació del model


models = {
    'cacd':{
        'inicial':'CACD/cacd_coral_basic/best_model.pt',
        'final':'CACD/best_model_cacd/best_model.pt'
        },
    'afad':{
        'inicial':'AFAD/afad-model1/best_model.pt',
        'final':'AFAD/best_model_afad/best_model.pt'
        }
    }


model34.load_state_dict(torch.load(models[DATASET][MODEL])) # load del model
path_altre=""


fotos = ['Ramon',
         'Infant',
         'Avi',
         'Bernat',
         'Adrià',
         'Noia AFAD',
         'Actor CACD',
         'Adarsh',
         'Sergio',
         "Altre"]

fotos_dir = ['probes_fotos/Ramon.jpg',
             'probes_fotos/Infant.jpg',
             'probes_fotos/Avi.jpeg',
             'probes_fotos/Bernat.jpg',
             'probes_fotos/Adria.jpg',
             'probes_fotos\AFAD_Noia.jpg',
             'probes_fotos\CACD_Actor.jpg',
             'probes_fotos\Adarsh.jpg',
             'probes_fotos\Sergio.jpg']

fotos_str = str(''.join([f' - {el} ({i+1})\n' for i, el in enumerate(fotos)]))
num = int(input(f"Escull l'imatge que vols probar\n{fotos_str} Escull:")) #\n -ramon(1)\n -bebe(2)\n -vell(3)\n -berni(4)\n"))

if num==10:
    path_altre=easygui.fileopenbox(title="Seleccionar imagen")
    fotos_dir.append(path_altre)

foto = Image.open(fotos_dir[num-1]) # escollir l'imatge de proba
custom_transform = transforms.Compose([transforms.Resize((128, 128)), 
                                       transforms.ToTensor()])

foto = custom_transform(foto)
foto = torch.unsqueeze(foto, 0)
probas = model34(foto)[1]

probas = probas.detach().numpy()
predicted_labels = np.sum(probas>0.5)

print(f"L'imatge '{fotos[num-1]}' té {predicted_labels+ADD_CLASS} anys")

