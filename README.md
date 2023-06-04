# Predicció d'edat en imatges facial
En aquest projecte presentat per l'assignatura de "Xarxes neuronals i aprenentatge profund" hem tractat el tema de l'aging, que consisteix a predir l'edat d'una persona a partir d'una imatge. No és un problema gens fàcil, ja que les característiques de la cara que indiquen l'envelliment van canviant al llarg dels anys. Per exemple, quan som joves, el fet de tenir o no barba, pot ajudar molt a l'hora de saber l'edat del subjecte. En canvi, a etapes més grans de la vida, hi ha altres detalls com el to de la pell, les arrugues o les taques que ens donen més informació sobre l'edat.
Els principals mètodes utilitzats han estat diferents arquitectures de CNN com ResNet, Mobilenet o Squeezenet. A part de les arquitectures, hem trobat que es pot fer servir tant regressió com classificació per predir els valors, però els mètodes més potents utilitzen regressió. El nostre starting point es basa en regressió emprant un mètode anomenat CORAL.

## CORAL
Per començar a fer el treball  s'ens va compartir un GitHub d'un treball sobre aging amb el mètode de CORAL (COnsistent RAnk Logits), el qual busca solucionar la inconsistència que tenen alguns regressors.
![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image10.png)

Aquesta inconsistència passa en els regressors, ja que no tenen en compte que la probabilitat de què la persona de la foto tingui l'edat k-1 sempre ha de ser major que tingui l'edat k. El model del CORAL ho soluciona utilitzant una llista de nivells on cada nivell ens diu que s'ha superat totes les edats inferiors. De manera que la probabilitat de tenir una edat k sempre serà més gran que la de tenir k+1.
Un punt fort del CORAL és que es pot aplicar a arquitectures convolucionals actuals, ja que només necessita fer uns pocs canvis en les últimes capes de les CNN.
Si suposem que un dataset té K edats diferents, CORAL es basa en la llista de nivells, on hi ha k-1 labels. Cada label indica si la persona supera el rank, per tant, tenim k-1 classificacions binàries.

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image14.png)

Per fer la predicció s'han de sumar totes les prediccions de cada rank i sumar-li 1, ja que estem treballant amb K-1 labels.

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image31.png)

Pel que fa a la loss function que utilitza el CORAL, és una cross entropy de les K-1 labels amb les que treballem. s(g(xi,W)+bk) es el predict de Xi, bk es el bias, Yi es el groundtruth, s es la sigmoide i delta(k) és el pes de la loss en el rang k, ja que depenent del rang importen més unes característiques o altres.

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image37.png)


####  Canvis en el codi:
- Modificar el DataLoader perque ens retorni també els nivells que hem mencionat abans.

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image22.png)

- Canviar l’arquitectura de la CNN (en el nostre cas d’una Resnet-34). Es canvia la fc per tenir un únic output i després es crea el output dels nivells.

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image17.png)

- El forward s’ha de canviar respecte als canvis que hem fet a l’arquitectura.

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image5.png)

- Afegim la loss del CORAL

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image24.png)

- Modificar el bucle de train respecte als canvis que hem fet al forward.

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image33.png)

## Datasets
Els nostres resultats s’han de basar en dos datasets diferents:
- AFAD (Asian Face Age Dataset) consisteix en 160K cares centrades de gent asiàtica. El rang d'edats va de 15 a 40 anys i també hi ha labels segons el sexe. A l'hora d'entrenar es va separar en 10% de test, 10% de validació i 80% d'entrenament.

- CACD (Cross-Age Celebrity Dataset) són també 160k cares, però ara són d'actors famosos de Hollywood. No hi ha label segons el sexe, únicament es guarda l'edat. Per cada actor hi ha entre unes 70 i 100 fotografies de diferents èpoques de la seva vida amb edats diferents. També hem fet la separació entre 10% de test, 10% de validació i 80% de train.

## Codi
Per començar a treballar hem utilitzat tot el codi del starting point i l'hem executat per tenir unes mètriques inicials amb les que comparar els resultats que anem obtenint.
En el GitHub dels creadors del paper del coral hi trobem tres models diferents amb una estructura similar. Un utilitza cross-entropy loss, un altre fa regressió ordinal i l'últim utilitza CORAL, ja explicat anteriorment.
Respecte a l'arquitectura els tres usen una Resnet-34, agafant el codi de la implementació de PyTorch. Amb relació als hiperparàmetres, fan servir l'optimizer Adam, 200 èpoques (que haurem de baixar, ja que estem limitats pel temps de les màquines), un learning rate inicial de 0.0005 i una inicialització dels pesos de la xarxa seguint una distribució normal.
Al final de cada una de les epochs, el codi guarda quin és el millor model comparant els resultats del conjunt Validation. Després de fer l'entrenament del model i la seva corresponent avaluació, s'agafa el "best_model" (el millor model segons el Validation) i s'avalua per guardar les seves mètriques finals.
Finalment, cal comentar que el batch size l'hem hagut d'anar canviant a causa del fet que hem anat fent servir les nostres targetes gràfiques personals. Com tenen memòries diferents a la de la targeta d'Azure, el valor s'ha hagut d'anar adaptant per evitar errors en l'execució.

## Resultats
Com es pot veure a la taula inferior, respecte el MAE i RMSE, els diferents models del starting point donen valors molt semblants entre ells. No hi ha un model que destaqui entre els altres. Igualment, centrarem el treball en el model CORAL, ja que és el que es prioritzava a l’Starting Point.

![imagen](https://github.com/DCC-UAB/xnap-project-ed_group_13/assets/103357825/185a3b29-1648-49d2-b499-ecef43431df6)

## Anàlisi del model CORAL
Començar comentant que tots els models inicials donaven overfit. En la gràfica de train i test es veu com la corba de color blau (test), s'estabilitza a partir de cert valor. En canvi, la corba taronja (train), continua baixant a mesura que avancen les epochs.
Amb els altres models del starting point passava el mateix.

![imagen](https://github.com/DCC-UAB/xnap-project-ed_group_13/assets/103357825/51f2a1d1-a794-4c22-a39a-cf6cfb735998)

Hem fet unes gràfiques per comparar les distribucions de les prediccions amb les del groundtruth. Serveixen per veure si hi ha alguna semblança en general o si falla en alguns anys en concret:

![imagen](https://github.com/DCC-UAB/xnap-project-ed_group_13/assets/103357825/f43769e1-06ed-40d5-a152-95a20fa353a5)

Es pot veure que en el CACD la distribució de la predicció no s’assembla gens a la del groundtruth i l’escala de la Y és força diferent. En el cas de l’AFAD pot semblar que la distribució és similar, però si ens fixem en l’escala de l’eix Y veiem com acaba tenint els problemes del CACD.
En els dos casos, el model tendeix a preveure més edats baixes que altes, sobretot tendint cap als 20/25 anys.

També vam fer una matriu de confusió per veure en quines edats els models encerten i fallen més:

![imagen](https://github.com/DCC-UAB/xnap-project-ed_group_13/assets/103357825/a0fa5179-1774-4fcc-9ea7-48c3a93ff7ad)

La primera figura és la confmat del model del CACD, on veiem que els punts de més brillantor, es troben al principi i al final de la diagonal principal. La zona intermitja falla més. Es pot veure perquè els punts estan molt més dispersos cap a la diagonal secundaria, pero tot i així es pot intuir la diagonal principal.

Sobre la segona, la de l’AFAD, es pot comentar el mateix que a la primera. En el principi i el final de la diagonal principal es veu que ha encertat més valors, i la zona del mig ha fallat més. Per tant, els punts es veuen més dispersos.

## Nou Dataset: UTK
Per sort per nosaltres, no són pocs els datasets de cares de persones per fer Age Recognition que existeixen a Internet. Nosaltres, després de trobar-lo mencionat en diversos papers, vam acabar agafat el dataset UTK Faces.
Aquest dataset està conformat per moltes menys fotos que els dos anteriors, tenint unes 20.000 fotos de persones entre 0 i 93 anys. Encara que nosaltres no ho hem tingut en compte, aquest dataset també guarda els atributs de raça i gènere per cada una de les fotografies.
Al tenir moltes menys fotografies, vam considerar convenient afegir més transformacions de Data Augmentation, ja que el RandomCrop del codi del Stating Point es podria quedar una mica insuficient. Per això, vam decidir afegir un RandomRotation i un RandomHorizontalFlip a les transformacions del Train. Igualment, també vam provar a entrenar-ho amb el data augmentation inicial per tenir una comparació:

#### Data Augmentation Starting Point:

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image30.png)

- MSE: 86.33
- RMSE: 9.27
- MAE: 6.66

Després de 50 epochs, els resultats del test son molt similars als resultats del dataset CACD però amb una mica menys d’overfit que aquests, cosa que demostra que el model funciona amb datasets fora del Starting Point.

#### Data Augmentation Extra:

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image6.png)

- MSE: 91.56
- RMSE: 9.54
- MAE: 6.69

Veiem que després de 50 epochs els resultats són extremadament semblants que sense utilitzar les noves transformacions, pero es redueix considerablement l’overfit, tenint només ara una diferència de 0.4 de MAE entre train i test.
El codi d’aquests models es troba a l’arxiu utk_coral.py

## Solucions al overfit
Utilitzant el model Coral com a referència, volem solucionar els diversos problemes que hem vist al model inicial, centrant-nos en l’overfit, fent proves amb diverses solucions que creiem que poden aconseguir una millora en el funcionament i en els resultats finals del mode:

### Data augmentation
Veient que en el UTKFaces l’addició de data augmentation va suposar una reducció de l’overfit, vam considerar afegir les mateixes transformacions en els dos datasets del Starting Point, un RandomRotation de 10 graus i un RandomHorizontalFlip:
#### AFAD:

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image11.png)

- MSE: 21.55
- RMSE: 4.64
- MAE: 3.38

El codi d’aquest model es troba a l’arxiu afad_coral.py, amb els canvis comentats.

#### CACD:

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image15.png)

- MSE: 59.36
- RMSE: 8.15
- MAE: 6.17

El codi d’aquest model es troba a l’arxiu cacd_coral.py, amb els canvis comentats.

Encara que els resultats milloren mínimament, no es suficient per dir que afegir Data Augmentation fa que el model predigui millor, però si que veiem una gran millora respecte a l’overfit comparant amb els models inicials, reduint-se dràsticament i mantenint-se constant a mesura que van avançant les epochs.

### Noves arquitectures
Una altra solució que ens vam plantejar va ser canviar l’arquitectura d’una ResNet-34 per mirar si donaven millors resultats. Vam decidir provar 4 arquitectures noves: dues resnet (una més densa i una altra menys densa que la 34), una Mobilenet i una Squeeznet. En aquest cas tots els resultats mostrat seràn del dataset AFAD, ja que en el cas de les dues últimes arquitectures mostrades, el model era incapaç d’entrenar, donant resultats inservibles.

#### Resnet-18:

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image39.jpg)

- MSE: 25.88
- RMSE: 5.08
- MAE: 3.69
	
Provant amb una ResNet menys densa, veiem com el dataset és massa gran per adaptar-se a aquesta arquitectura, resultant en un overfit molt més gran que el ja mencionat a la ResNet-34.
El codi d’aquest model es troba a l’arxiu afad_coral_resnet18.py

#### Resnet-50:

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image34.png)

- MSE: 26.01
- RMSE: 5.10
- MAE: 3.73

Amb una xarxa més densa com es la ResNet-50, ens donen resultats molt similars que el model original, tant en mètriques com en overfit. Al tenir un temps d’execució major que el original i no millorar els resultats, no ens surt a compte utilitzar aquesta arquitectura en un hipotètic model final.
El codi d’aquest model es troba a l’arxiu afad_coral_resnet50.py

#### Mobilenet:

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image25.png)

- MSE: 27
- RMSE: 5.19
- MAE: 3.77

Amb un model més lleuger i més eficient com és la mobilenet, torna a donar resultats similars que a la ResNet-34, pero sent la arquitectura que menys overfit té en 35 epochs entre totes les que hem probat, pero sense arribar a quedar-se constant. A més, no les mètriques no fan una curva, sinò que baixen en les primeres epochs i s’estanquen fins que tornen a baixar en un punt més avançat de l’execució.
Per molt que funcioni millor, al no anar amb el dataset CACD no ho podem utilitzar per un model final.
El codi d’aquest model es troba a l’arxiu afad_coral_mobilenet.py


#### Squeezenet:

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image16.png)

- MSE: 40.51
- RMSE: 6.36
- MAE: 4.91

Squeezenet és una xarxa també lleugera i el model té un comportament similar a la Mobilenet, pero en aquest cas el model s’estanca completament i oscil·la entre els mateixos valors durant tot el temps que està activa, sense baixar com passa a la Mobilenet, valors que són els pitjors de totes les arquitectures provades.
El codi d’aquest model es troba a l’arxiu afad_coral_squeezenet.py

### Transfer Learning
Per tal de millorar l’inicialització dels pesos del model original, les quals les fa seguint una distribució normal, vam pensar en aprofitar-nos de que ja estava fent servir una arquitectura pre-existent (Resnet-34), i gràcies a la funció load_state_dict() de pytorch, podem precarregar els pesos de una resnet ja entrenada. En el cas de Pytorch, aquest model està entrenat en el dataset de IMAGENET1K de classificació d’imatges. Fent servir això, podriem aconseguir una millora en la curva d’aprenentatge i uns millors resultats finals en el model. Per tant, vam provar de fer servir els dos mètodes de transfer learning: Feature Extraction y Fine Tuning.

#### Feature Extraction
Aquí varem probar a carregar els pesos corresponents, congelar totes les capes menys l’última capa lineal, i entrenar el model (encara que realment nomès estem entrenant l’última capa) amb dataset cacd. Inicialment, per veure si aquesta idea funcionaba suficientment bé, vam fer servir la Cross Entropy loss, ja que la coral loss afegeix capes extras a la arquitectura, i creiem que no seria un bon exemple per tal de comprobar la eficacia d’aquest métode. Al entrenar el model, però, la loss no baixava i el model no aprenia res.

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image18.png)

Això és degut a que els pesos importants són els de un model de classificació d’imatges, no de predicció d’edats. Així que no vam continuar explorant aquesta línia de millora més, i vam decidir centrar-nos en aplicar fine tuning.
El codi d’aquest model es troba a l’arxiu cacd_ce_transfer.py

#### Fine tuning
L’implementació del fine tuning és igual que feature extraction, però aquest cop no congelem cap capa. En aquesta ocasió si que vam decidir aplicar fine tuning sobre el model amb la loss del coral, inicialitzant les capes de l’arquitectura Resnet-34 amb els pesos importats, i les capes afegides noves degut al coral les inicialitzarem com feiem previament en el model original. Aquest cop, el nostre model si que va donar bons resultats:

#### AFAD:

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image32.png)

- MSE: 21.95
- RMSE: 4.68
- MAE: 3.41

#### CACD:

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image2.png)

- MSE: 64.76
- RMSE: 8.04
- MAE: 5.93

Com observem en els resultats, aquests són millors que en el model original, tot i que no millora molt l’overfit. Per tant, tot i que hem vist que pot ser un bon métode alternatiu d’inicialització dels pesos, no és suficient per a aconseguir el nostre objectiu de generalitzar millor el model.
El codi d’aquest model es troba als arxius cacd_coral_transfer.py i afad_coral_transfer.py.

### Canvi d’optimizer
Un últim canvi que vam voler probar va ser el de cambiar l’optimizer utilitzat, per un altre on també tinguessim un learning rate variable, tot i que sabiem que l’Adam en aquest sentit ja era força efectiu. Per tal de buscar un canvi més significatiu, en comptes de probar algoritmes com l’Adagrad o el RMSProp, que són força similars basats en el mateix concepte d’adaptar el learning rate en funció dels valors del gradient, vam voler probar a aplicar un Stochastic Gradient Descent, aplicant diferents funcions de scheduler per al learning rate, per a veure com podia afectar als resultats cadascuna. 

Les dues funcions de scheduler probades van ser la Cosine Annealing Warm Restarts

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image28.png)

I la cyclicLR:

![alt text](https://github.com/DCC-UAB/xnap-project-ed_group_13/blob/main/images/image13.png)

Aquests models, però, no van prosperar molt ja que molts d’ells no entrenaben, comencaven a donar valors nulls en mig del entrenament, i els que arrivaben a entrenar una mica, donaven resultats poc coherents (la loss augmentaba més del compte) o massa dolents en comparació, així que vam decidir descartar aquest approach en favor d’intentar i centrar-nos en altres metodologies per a millorar el model. Creiem que aquests resultats incoherents surgeixen ja que, al no haver normalitzat les dades, el valor de la loss creix desmesuradament en moments de l’entrenament, i el codi deixa de funcionar correctament degut a falles amb la precisió de les variables flotants. 
El codi d’aquest model es troba a l’arxiu safad-coral-copy schedule.py

## Model final
Després de tots els canvis amb els que hem anat experimentant i hem provat al llarg d’aquest projecte, finalment ens hem quedat amb els que creiem, són els més significatius i que unificats en un mateix model tindràn el millor resultat possible que podem aconseguir. Aquest model final, basat en el codi original, està compost de 3 factors principals:

- Coral loss :  de les 3 losses proposades, aquesta és la més efectiva i la que ens dona els millors resultats, per tant, la que creiem més adient per aplicar a la nostra solució.

- Data augmentation : tot i que el model original aplica certes tècniques de data augmentation a les imatges, com hem comentat previament, aplicar certes operacions adicionals ajuda molt a reduir l’overfit.

- Resnet-34 amb Fine-Tuning : segons els resultats obtinguts, creiem que una Resnet-34 inicialitzant els pessos amb transfer learning, i aplicant fine-tuning, tot i que no redueix l’overfit, és un model amb el que obtenim els millors resultats.

Aplicant tots aquest canvis sobre un mateix model obtenim els següents resultats:

#### AFAD:

![imagen](https://github.com/DCC-UAB/xnap-project-ed_group_13/assets/103357825/237bfa55-0487-4f55-98cc-bf30d062f01b)

- MSE: 19.27
- RMSE: 4.39
- MAE: 3.19

#### CACD:

![imagen](https://github.com/DCC-UAB/xnap-project-ed_group_13/assets/103357825/60034ffe-77ab-4227-9baf-1600eb92cb3c)

- MSE: 59.60
- RMSE: 7.72
- MAE: 5.73

Si comparem aquests resultats amb els valors que obteniem amb el valor original, observem una millora en en els valors del MAE i MSE, i una forta reducció del overfit.
També podem fer una comparació de les distribucions de les prediccions. A la esquerra el model inicial, i a la dreta el model final.
Per el dataset CACD:

![imagen](https://github.com/DCC-UAB/xnap-project-ed_group_13/assets/103357825/800ccc32-372e-442c-8030-6a70ea8fc655)

I per el dataset AFAD:

![imagen](https://github.com/DCC-UAB/xnap-project-ed_group_13/assets/103357825/56718513-bcb2-40c9-808f-d9d5098f5a10)

Tot i que si observem, especialment en les matrius de confusió, que les prediccions estan una mica millor balancejades, la diferència no és gaire notable, i el model seguix patint els mateixos problemas que perjudicaban al model inicial en aquest sentit.
Conclusions i millores

Finalment, creiem que hem fet un bon treball asolint els nostre objectiu inicial, el qual era entendre bé el problema i l’starting point proposat, i procurar de millorar-ho el màxim possible. Tot i que els resultats no són significativament millors, si que hem aconseguit reduir en gran part l’overfit que presentava el model inicial, el qual ha sigut el nostre principal enfocament a l’hora de buscar solucions. 

Alhora que buscavem millorar les prediccions, hem aprés també a entendre millor el funcionament de un model en un entorn més real i pràctic, i ha haver de buscar, probar, i raonar diferentes metodologies que hem aplicat al codi, i saber entendre perque funcionaven o no. 

Com a millores que proposem per avançar aquest treball, una opció que creiem molt important i que no hem provat degut a falta d’hores en les màquines virtuals, és la de entrenar el model amb 200 epochs, com fa l’starting point, ja que creiem que el model arrivaria a tenir millors resultats (tot i que no desmesuradament millors).

## Execució del codi per fer prediccions
El codi probes_fotos.py és un codi que ens deixa fer prediccions d’edat amb diferents models. Després d’executar l’arxiu .py, hem de seguir aquests passos:
- Primer escollirem el dataset amb el que s’haurà entrenat el model: AFAD o CACD.
- Ara decidirem si volem utilitzar el model inicial o el final.
- Escollim la imatge entre les que hem guardat al GitHub per fer proves o amb una del propi disc:

![imagen](https://github.com/DCC-UAB/xnap-project-ed_group_13/assets/103357825/96b5a89d-7632-4481-b3f0-087a48053034)


- I ens retornarà el resultat:

![imagen](https://github.com/DCC-UAB/xnap-project-ed_group_13/assets/103357825/9fc30d09-9243-4f8a-a8fd-0c4535bd4339)

En el cas de l’entrega al campus, al no poder penjar els pesos, no es podrà exectuar.

## Contributors

Bernat Medina Perez - 1606505@uab.cat

Adrià Martinez Vega - 1604392@uab.cat

Sergio Trigueros Hevia - 1605983@uab.cat

Xarxes Neuronals i Aprenentatge Profund Grau de Enginyeria de Dades, UAB, 2023
