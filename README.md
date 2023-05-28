[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11122259&assignment_repo_type=AssignmentRepo)
# Predicció d'edat en imatges facials

En aquest projecte, a partir del repositori de Raschka Research Group (https://github.com/Raschka-research-group/coral-cnn), crearem i jugarem amb diferents models de Deep learning amb l'objectiu de entendre el problema i els models del respositori original, crear la millor solució possible i investigar quins factors poden tenir un major impacte en aquest problema.

## Estructura del codi

En la carpeta models es troba el codi de tots els models creats en aquest repositori, que llegeixen la informació dels datasets def la carpeta "Starting point/datasets". Els datasets, degut al seu gran tamany, no es troben al repositori, així que caldrà modificar la variable IMAGE_PATH de cada codi del model amb l'ubicació del dataset en la màquina local si es vol executar el codi per entrenar els models.


## Models

A la carpeta models hi haurà una sub-carpeta per a cadascun dels datasets utilitzats. Els noms de cada un dels scripts està compost de la següent manera:

"dataset" - "loss" - "model".py

On:

Dataset - és el dataset utilitzat per a entrenar a aquell model
Loss - La loss o estrategia utilitzada per a entrenar el model (coral, ce, oridnal)
Model - Si escau, el canvi en la arquitectura o en el model utilizat en aquell script


## Contributors
Bernat Medina Perez - 
Adrià Martinez Vega - 
Sergio Trigueros Hevia - 1605983@uab.cat


Xarxes Neuronals i Aprenentatge Profund
Grau de Enginyeria de Dades, 
UAB, 2023
