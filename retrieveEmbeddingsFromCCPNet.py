'''0427 for RobMOTS
Add PointTrack embeddings to all segmentation results
'''
import os, sys
import torch
from PIL import Image
import numpy as np
from models import get_model
from config import *
os.chdir(rootDir)
from config_mots import *
from file_utils import *
from torchvision.transforms import functional as TF
import pycocotools.mask as maskUtils
from utils import transforms as my_transforms


def saveLines(curr_content, curr_txt):
    with open(curr_txt, 'w') as f:
        for line in curr_content:
            f.write(line + '\n')


# trainval
# SEQ_IDS = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]] + ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
SEQ_IDS = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
TIMESTEPS_PER_SEQ = {"0000": 154, "0001": 447, "0002": 233, "0003": 144, "0004": 314, "0005": 297, "0006": 270,
                     "0007": 800, "0008": 390, "0009": 803, "0010": 294, "0011": 373, "0012": 78, "0013": 340,
                     "0014": 106, "0015": 376, "0016": 209, "0017": 145, "0018": 339, "0019": 1059, "0020": 837}
imgRoot = os.path.join(kittiRoot, "images")
# txtRoot = os.path.join(kittiRoot, "instances_gt_txt")
# person_save_dir = os.path.join(kittiRoot, 'person_trainval_gt_w_emb')
# car_save_dir = os.path.join(kittiRoot, 'car_trainval_gt_w_emb')
txtRoot = os.path.join(kittiRoot, "valset_results/valset_result_CCP_ST")
person_save_dir = os.path.join(kittiRoot, 'person_trainval_CCP_w_emb')
car_save_dir = os.path.join(kittiRoot, 'car_trainval_CCP_w_emb')
# # test
# SEQ_IDS = ["%04d" % idx for idx in range(29)]
# TIMESTEPS_PER_SEQ = {'0000': 465, '0015': 701, '0017': 305, '0003': 257, '0001': 147, '0018': 180, '0005': 809,
#                           '0022': 436, '0021': 203, '0023': 430, '0012': 694, '0008': 165, '0009': 349, '0020': 173,
#                           '0016': 510, '0013': 152, '0004': 421, '0028': 175, '0024': 316, '0019': 404, '0026': 170,
#                           '0007': 215, '0014': 850, '0025': 176, '0027': 85, '0011': 774, '0010': 1176, '0006': 114,
#                           '0002': 243}
# imgRoot = os.path.join(kittiRoot, 'testing/image_02/')
# # txtRoot = os.path.join(kittiRoot, "testset_results/FGNet")
# # person_save_dir = os.path.join(kittiRoot, 'person_testset_FGNet_w_emb')
# # car_save_dir = os.path.join(kittiRoot, 'car_testset_FGNet_w_emb')
# txtRoot = os.path.join(kittiRoot, "testset_results/CCP")
# person_save_dir = os.path.join(kittiRoot, 'person_testset_CCP_w_emb')
# car_save_dir = os.path.join(kittiRoot, 'car_testset_CCP_w_emb')


# Load models for cars and pedestrians
device = 'cuda:0'
person_args = eval('person_mots_ccpnet').get_args()
person_model = get_model(person_args['model']['name'], person_args['model']['kwargs'])
person_model.init_output(person_args)
person_model.eval()
person_model.load_from(person_args['resume_path'])
person_model.to(device)
car_args = eval('car_mots_ccpnet').get_args()
car_model = get_model(car_args['model']['name'], car_args['model']['kwargs'])
car_model.init_output(car_args)
car_model.eval()
car_model.load_from(car_args['resume_path'])
car_model.to(device)


mkdir_if_no(person_save_dir)
mkdir_if_no(car_save_dir)
person_id, car_id=2, 1


def pad_array(imgs, pad_size):
    w, h = imgs[0].size
    padding = (0, 0, pad_size[1] - w, pad_size[0] - h)
    pad_mask = np.zeros(pad_size,
                        dtype=np.uint8)  # a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
    pad_mask[h:, w:] = 1
    return [TF.pad(el, padding) for el in imgs] + [Image.fromarray(pad_mask)]


def pre_process(img_path, idDict, pad_size = (384, 1248)):
    sample = {}
    image = Image.open(img_path)
    sample['im_name'] = img_path
    w, h = image.size
    sample['im_shape'] = np.array([image.size])
    # load instanceMap according to idDict
    inst_np, label_np = np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
    instItems = []
    for inst_id, vals in enumerate(idDict):
        instItems.append(vals['items'])
        inst_mask = maskUtils.decode(vals)
        inst_np[inst_mask.astype(np.bool)] = inst_id+1
        label_np[inst_mask.astype(np.bool)] = 1
    instance, label = Image.fromarray(inst_np), Image.fromarray(label_np)
    # if not self.type == 'val':
    image, instance, label, pad_mask = pad_array([image, instance, label], pad_size)
    sample['image'] = image
    sample['instance'] = instance
    sample['label'] = label
    sample['pad_mask'] = pad_mask
    return transform(sample)


def process_video(id, idDict, model):
    image_root = os.path.join(imgRoot, id)

    resDict = {}
    for k, v in idDict.items():
        # decode each png
        img_path = os.path.join(image_root, '%06d' % int(float(k)) + '.png')
        W, H = Image.open(img_path).size
        instance_map = np.zeros((H, W)).astype(np.uint8)

        # get embedding for each mask
        sample = pre_process(img_path, v)
        ims = sample['image'].unsqueeze(0).to(device)
        instances = sample['instance'].to(device)
        class_labels = sample['label'].to(device)
        output, embeddings = model.forward(ims, seg=True)
        (w, h) = sample['im_shape'][0]
        embeddings, instance_maps = embeddings[:, :, :h.item(), :w.item()], instances[:, :h.item(), :w.item()]
        instEmbeds, instMasks, instUids = model.construct_targets_val([embeddings], [instance_maps])

        for vind, v_ in enumerate(v):
            v_['emb'] = instEmbeds[vind].detach().cpu().numpy().tolist()
        resDict[k] = v
    return resDict

transform = my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label', 'pad_mask'),
                        'type': (torch.FloatTensor, torch.LongTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                }])
for id in SEQ_IDS:
    print(id)
    # noPersonIds, noCarIds = NoPersonFrames[id], NoCarFrames[id] # for filtering bad cases
    noPersonIds, noCarIds = [], []
    label_root = os.path.join(txtRoot, id + '.txt')
    person_save_loc = os.path.join(person_save_dir, id + '.pkl')
    car_save_loc = os.path.join(car_save_dir, id + '.pkl')

    nums = TIMESTEPS_PER_SEQ[id]
    personidDict, caridDict = {}, {}
    for i in range(nums):
        personidDict[str(i)] = []
        caridDict[str(i)] = []

    with open(label_root, 'r') as f:
        car_person_content = f.readlines()
    res_content = []

    for line in car_person_content:
        ann_str = line.encode().decode("utf-8")
        vals = ann_str.split()
        if vals[2] == str(person_id):
            personidDict[vals[0]].append({'size': (int(float(vals[3])), int(float(vals[4]))), 'counts': vals[5], 'class': person_id, 'items': vals})

        if vals[2] == str(car_id):
            caridDict[vals[0]].append({'size': (int(float(vals[3])), int(float(vals[4]))), 'counts': vals[5], 'class': car_id, 'items': vals})

    save_pickle2(person_save_loc, process_video(id, personidDict, person_model))
    save_pickle2(car_save_loc, process_video(id, caridDict, car_model))
