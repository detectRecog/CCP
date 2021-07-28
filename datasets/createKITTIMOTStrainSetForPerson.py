from file_utils import *
from config import kittiRoot
from random import shuffle
from PIL import Image
import numpy as np
# select frames with persons, randomly select 50%, 其他随机选取20%

OBJ_ID = 2
mots_instance_root = os.path.join(kittiRoot, 'instances')
SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
TIMESTEPS_PER_SEQ = {"0000": 154, "0001": 447, "0002": 233, "0003": 144, "0004": 314, "0005": 297, "0006": 270,
                     "0007": 800, "0008": 390, "0009": 803, "0010": 294, "0011": 373, "0012": 78, "0013": 340,
                     "0014": 106, "0015": 376, "0016": 209, "0017": 145, "0018": 339, "0019": 1059, "0020": 837}

mots_persons = []
for subdir in SEQ_IDS_TRAIN:
    instance_list = sorted(make_dataset(os.path.join(mots_instance_root, subdir), suffix='.png'))
    instance_list = ['/'.join(el.split('/')[-2:]) for el in instance_list]
    for i in instance_list:
        mots_persons.append(i)

print('trainSet %s samples' % len(mots_persons))


def hasPerson(suffix):
    instance_path = os.path.join(mots_instance_root, suffix)
    instance = Image.open(instance_path)

    instance_np = np.array(instance, copy=False)
    object_mask = np.logical_and(instance_np >= OBJ_ID * 1000, instance_np < (OBJ_ID + 1) * 1000)
    ids = np.unique(instance_np[object_mask])
    return ids.shape[0] > 0


mots_val_persons, mots_val_noPersons = [], []
for subdir in SEQ_IDS_VAL:
    instance_list = sorted(make_dataset(os.path.join(mots_instance_root, subdir), suffix='.png'))
    instance_list = ['/'.join(el.split('/')[-2:]) for el in instance_list]
    for i in instance_list:
        if hasPerson(i):
            mots_val_persons.append(i)
        else:
            mots_val_noPersons.append(i)
shuffle(mots_val_persons)
shuffle(mots_val_noPersons)

print('val has person:', len(mots_val_persons), 'val no persons:', len(mots_val_noPersons))
mots_persons += mots_val_persons[:int(len(mots_val_persons)*0.6)]
mots_persons += mots_val_noPersons[:int(len(mots_val_noPersons)*0.3)]
print('trainSet+valSet %s samples' % len(mots_persons))
save_pickle2(os.path.join(kittiRoot, 'mots_person_train5.pkl'), mots_persons)
