import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from afad_coral_copy import resnet34 






'''
# Per carregar un model primer has de crear l'arquitectura en questió i despres carregues el fitxer .pt de torch amb un load
'''

DATASET = 'afad' # seleccionar el dataset del model [cacd/afad]
MODEL = 'final' # seleccionar quin model es vol [inicial/final]


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
        'inicial':'cacd_coral_basic/best_model.pt',
        'final':''
        },
    'afad':{
        'inicial':'afad-model1/best_model.pt',
        'final':'best_model_afad/best_model.pt'
        }
    }


model34.load_state_dict(torch.load(models[DATASET][MODEL])) # load del model


foto_cacd = Image.open('../../../shared_datasets/CACD/centercropped/jpg/CACD2000/15_Chris_Brown_0007.jpg')
foto_afad = Image.open('../../../shared_datasets/AFAD/orig/tarball/AFAD-Full/23/111/330-0.jpg')
foto_ramon = Image.open('../../../shared_datasets/probes/proba_foto_ramon.jpg') # fotos de proba
foto_bebe = Image.open('../../../shared_datasets/probes/q_tonto_jajaj.jpg') # fotos de proba
foto_viejo = Image.open('../../../shared_datasets/probes/abueloo.jpeg') # fotos de proba
foto_berni = Image.open('../../../shared_datasets/probes/proba_berni.jpg')
foto = ... # escollir l'imatge de proba
custom_transform = transforms.Compose([transforms.Resize((128, 128)), 
                                       transforms.ToTensor()])

foto = custom_transform(foto)
foto = torch.unsqueeze(foto, 0)
probas = model34(foto)[1]
# predict_levels = probas > 0.5
# print(predict_levels)
probas = probas.detach().numpy()
predicted_labels = np.sum(probas>0.5)
print(predicted_labels+ADD_CLASS)
print(probas)

