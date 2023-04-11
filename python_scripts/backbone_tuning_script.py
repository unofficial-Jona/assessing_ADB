from backbone_tuning_supervised import train_one_epoch_supervised
from backbone_tuning_unsupervised import train_one_epoch_unsupervised

print('start supervised training')
for _ in range(5):
    train_one_epoch_supervised()

'''
print('start unsupervised training')
train_one_epoch_unsupervised()
'''