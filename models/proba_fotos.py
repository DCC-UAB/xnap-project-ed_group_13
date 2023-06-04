import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from AFAD.afad_coral_copy import resnet34 






'''
# Per carregar un model primer has de crear l'arquitectura en questió i despres carregues el fitxer .pt de torch amb un load
'''

DATASET = 'cacd' if int(input('seleccionar el dataset del model cacd(1)/afad(2)\n')) == 1 else 'afad'

MODEL = 'inicial' if int(input('seleccionar quin model es vol inicial(1)/final(2)\n')) == 1 else 'final'


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



fotos = ['ramon',
         'bebe',
         'vell',
         'berni',
         'adri',
         'noia AFAD',
         'actor CACD',
         'adarsh',
         'sergio']
fotos_dir = ['probes_fotos/proba_foto_ramon.jpg',
             'probes_fotos/q_tonto_jajaj.jpg',
             'probes_fotos/abueloo.jpeg',
             'probes_fotos/proba_berni.jpg',
             'probes_fotos/aadri.jpg',
             'probes_fotos\afad_noia_jove.jpg',
             'probes_fotos\cacd_mcgregor.jpg',
             'probes_fotos\darsh.jpg',
             'probes_fotos\sergioo.jpg']
fotos_str = str(' '.join([f' -{el} ({i+1})\n' for i, el in enumerate(fotos)]))
num = int(input(f"Escull l'imatge que vols probar\n{fotos_str}")) #\n -ramon(1)\n -bebe(2)\n -vell(3)\n -berni(4)\n"))

foto = Image.open(fotos_dir[num-1]) # escollir l'imatge de proba
custom_transform = transforms.Compose([transforms.Resize((128, 128)), 
                                       transforms.ToTensor()])

foto = custom_transform(foto)
foto = torch.unsqueeze(foto, 0)
probas = model34(foto)[1]

probas = probas.detach().numpy()
predicted_labels = np.sum(probas>0.5)
print(f"L'imatge '{fotos[num-1]}' té {predicted_labels+ADD_CLASS} anys")

