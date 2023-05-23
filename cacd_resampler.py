import pandas as pd

train = pd.read_csv('cacd_train.csv')
train.sample(n=10000,axis=0,random_state = 420).reset_index().to_csv('cacd_train_sample.csv')

test = pd.read_csv('cacd_test.csv')
test.sample(n=256,axis=0,random_state = 420).reset_index().to_csv('cacd_test_sample.csv')

valid = pd.read_csv('cacd_valid.csv')
valid.sample(n=256,axis=0,random_state = 420).reset_index().to_csv('cacd_valid_sample.csv')

