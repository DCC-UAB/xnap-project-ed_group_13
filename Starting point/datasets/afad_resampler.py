import pandas as pd

train = pd.read_csv('afad_train.csv')
train.sample(n=10000,axis=0,random_state = 420).reset_index().to_csv('afad_train_sample.csv')

test = pd.read_csv('afad_test.csv')
test.sample(n=1000,axis=0,random_state = 420).reset_index().to_csv('afad_test_sample.csv')

valid = pd.read_csv('afad_valid.csv')
valid.sample(n=1000,axis=0,random_state = 420).reset_index().to_csv('afad_valid_sample.csv')

