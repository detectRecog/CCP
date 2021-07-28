"""
for reproduce
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os, sys
import shutil
import time
from config import *
os.chdir(rootDir)
from matplotlib import pyplot as plt
from tqdm import tqdm
from config_mots import *
from criterions.mots_seg_loss import *
from criterions.my_loss import *
from datasets import get_dataset
from models import get_model
from utils.utils import AverageMeter, ClusterSeedCls, Logger, Visualizer, ClusterSeedClsWithFilter, ClusterSeedClsWithFilter0907
from file_utils import remove_key_word
from models.radam import RAdam
import subprocess
from file_utils import save_pickle2

# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False
config_name = sys.argv[1]

args = eval(config_name).get_args()

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")
# clustering
# cluster = ClusterSeedClsWithFilter0907()
cluster = ClusterSeedClsWithFilter()
print('avg_seed: ', args['avg_seed'],
      ', threshold: ', args['threshold'],
      ', min_seed: ', args['min_seed_thresh'],
      ', inst_ratio: ', args['inst_ratio'],
      ', dist_thresh: ', args['dist_thresh'],
      ', min_pixel: ', args['min_pixel'],)

# Visualizer
visualizer = Visualizer(('image', 'pred', 'sigma', 'seed'))

# Logger
logger = Logger(('train', 'val', 'iou'), 'loss')

# train dataloader
train_dataset = get_dataset(
    args['train_dataset']['name'], args['train_dataset']['kwargs'])
train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# val dataloader
val_dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs'])
val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# set model
model = get_model(args['model']['name'], args['model']['kwargs'])
model.init_output(args['loss_opts']['n_sigma'])
model = torch.nn.DataParallel(model).to(device)

# set criterion
criterion = eval(args['loss_type'])(**args['loss_opts'])
criterion = torch.nn.DataParallel(criterion).to(device)
criterionVal = eval(args['loss_type'])(foreground_weight=10, to_center=args['loss_opts']['to_center'], eval=True)
# if args['train_dataset']['kwargs']['type']=='crop':
#     criterionVal = eval(args['loss_type'])(foreground_weight=50, to_center=args['loss_opts']['to_center'], eval=True) # 统一val时候loss的计算
# elif 'person' in args['train_dataset']['name']:
#     criterionVal = eval(args['loss_type'])(foreground_weight=50, to_center=args['loss_opts']['to_center'], eval=True)
# else:
#     criterionVal = eval(args['loss_type'])(foreground_weight=50, to_center=args['loss_opts']['to_center'], eval=True)
criterionVal = torch.nn.DataParallel(criterionVal).to(device)

# resume
start_epoch = 0
best_iou = 0
best_seed, best_val = 10, 100
best_seg = 0.0
min_seg = 0.7
max_disparity = args['max_disparity']
temp_state_file = os.path.join(args['save_dir'], 'temp.pth')
if 'resume_path' in args.keys() and args['resume_path'] is not None and os.path.exists(args['resume_path']):
    print('Resuming model from {}'.format(args['resume_path']))
    state = torch.load(args['resume_path'])
    if 'start_epoch' in args.keys():
        start_epoch = args['start_epoch']
    elif 'epoch' in state.keys():
        start_epoch = state['epoch'] + 1
    else:
        start_epoch = 1
    # best_iou = state['best_iou']
    for kk in state.keys():
        if 'state_dict' in kk:
            state_dict_key = kk
            break
    new_state_dict = state[state_dict_key]
    if not 'state_dict_keywords' in args.keys():
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except:
            print('resume checkpoint with strict False')
            model.load_state_dict(new_state_dict, strict=False)
    else:
        new_state_dict = remove_key_word(state[state_dict_key], args['state_dict_keywords'])
        model.load_state_dict(new_state_dict, strict=False)
        print('resume checkpoint with strict False')
    try:
        logger.data = state['logger_data']
    except:
        pass

# set optimizer
if 'decode_only' in args.keys() and args['decode_only']:
    print('finetune decode_only')
    optimizer = torch.optim.Adam(list(model.module.decoders[0].parameters())+
                                 list(model.module.decoders[1].parameters())+
                                 list(model.module.decoders[2].parameters()), lr=args['lr'], weight_decay=1e-4)
else:
    optimizer = RAdam(model.parameters(), lr=args['lr'], weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), args['lr'], momentum=0.9)


def lambda_(epoch):
    return pow((1 - ((epoch) / args['n_epochs'])), 0.9)


if 'milestones' in args.keys():
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['milestones'], gamma=args['gamma'] if 'gamma' in args.keys() else 0.1)
else:
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_, )


def train(epoch):
    global nan_list, nan_state, model, device
    torch.cuda.empty_cache()
    # define meters
    loss_meter = AverageMeter()
    loss_seed_meter = AverageMeter()

    # put model into training mode
    model.train()
    if 'fix_bn' in args.keys() and args['fix_bn']:
        print('BN Fixed!')
        # freeze bn if need
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))

    for i, sample in enumerate(tqdm(train_dataset_it,position=0, leave=True)):
        if i % 500==1:
            torch.save(model.state_dict(), temp_state_file)
        ims = sample['image']
        instances = sample['instance'].squeeze(1)
        class_labels = sample['label'].squeeze(1)

        output = model(ims)
        seed_w = sample['seed_w'].squeeze(1) if 'seed_w' in sample.keys() else None
        loss, seed_loss = criterion(output, instances, class_labels, **args['loss_w'], seed_w=seed_w, show_seed=True)

        loss = loss.mean()
        seed_loss = seed_loss.mean()

        if not torch.isnan(loss):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())
            loss_seed_meter.update(seed_loss.item())
        else:
            raise NotImplementedError

    return loss_meter.avg, loss_seed_meter.avg


def val(epoch):
    # define meters
    loss_meter, iou_meter, loss_seed_meter = AverageMeter(), AverageMeter(), AverageMeter()

    # put model into eval mode
    model.eval()
    clusterVal = torch.nn.DataParallel(cluster).to(device)

    if not os.path.isdir(args['seg_dir']):
        os.mkdir(args['seg_dir'])
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it,position=0, leave=True)):
            if iou_meter.count[0] > 10 and iou_meter.avg < min_seg:
                break
            ims = sample['image']
            instances = sample['instance'].squeeze(1)
            class_labels = sample['label'].squeeze(1)
            im_names = sample['im_name']

            output = model(ims)
            loss, focal_loss = criterionVal(output, instances, class_labels, **args['loss_w'], iou=True, iou_meter=iou_meter, show_seed=True)
            loss_seed_meter.update(focal_loss.mean().item())
            loss_meter.update(loss.mean().item())
            sizes = sample['im_shape'].squeeze(1)
            for ind, (w,h) in enumerate(sizes):
                output[ind, :, w:, h:] = 0
            # output_ = output[ind, :, :h, :w]
            instance_maps = clusterVal(output, threshold=args['threshold'],
                                                    min_pixel=args['min_pixel'],
                                                    min_inst_pixel=args[
                                                        'min_inst_pixel'] if "min_inst_pixel" in args.keys() else
                                                    args['min_pixel'],
                                                    min_seed_thresh=args[
                                                        'min_seed_thresh'] if "min_seed_thresh" in args.keys() else 0.5,
                                                    dist_thresh=args[
                                                        'dist_thresh'] if "dist_thresh" in args.keys() else 0.5,
                                                    inst_ratio=args['inst_ratio'] if "inst_ratio" in args.keys() else 0.5,
                                                    n_sigma=args["n_sigma"] if "n_sigma" in args.keys() else 2,
                                       avg_seed=args["avg_seed"] if "avg_seed" in args.keys() else 0.0)
            for ind, (w, h) in enumerate(sizes):
                video, frameCount = im_names[ind].split('/')[-2:]
                frameCount = int(float(frameCount.split('.')[0]))
                base = video + '_' + str(frameCount) + '.pkl'
                instance_map_np = instance_maps[ind][:h, :w].cpu().numpy()
                save_pickle2(os.path.join(args['seg_dir'], base), instance_map_np)

        if iou_meter.avg > min_seg:
            # eval on args['save_dir']
            if 'person' in args['save_dir']:
                p = subprocess.run([pythonPath, "-u", "test_tracking.py", 'person_test_tracking_val'], stdout=subprocess.PIPE, cwd=rootDir)
            else:
                p = subprocess.run([pythonPath, "-u", "test_tracking.py", 'car_test_tracking_val'], stdout=subprocess.PIPE, cwd=rootDir)

            pout = p.stdout.decode("utf-8")
            if 'person' in args['save_dir']:
                class_str = "Evaluate class: Pedestrians"
            else:
                class_str = "Evaluate class: Cars"
            pout = pout[pout.find(class_str):]
            print(pout[pout.find('all   '):][6:126].strip())
            acc = pout[pout.find('all   '):][6:26].strip().split(' ')[0]
        else:
            acc=0.0

    try:
        return iou_meter.avg, float(acc), loss_seed_meter.avg, loss_meter.avg
    except:
        return iou_meter.avg, 0.0, loss_seed_meter.avg, loss_meter.avg


def save_checkpoint(state, is_best, val_iou, best_seg, val_seed_loss, train_loss, val_loss, is_val_lowest=False, is_lowest=False, is_seed_lowest=False, name='checkpoint.pth'):
    print('=> saving checkpoint')
    if 'save_name' in args.keys():
        file_name = os.path.join(args['save_dir'], args['save_name'])
    else:
        file_name = os.path.join(args['save_dir'], name)
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_iou_model.pth' + '{:.4f}'.format(val_iou)))
    if is_seed_lowest:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_focal_model.pth' + '{:.8f}'.format(val_seed_loss) + '_' + '{:.4f}'.format(train_loss)))
    if is_val_lowest:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_val_model.pth' + '{:.8f}'.format(val_loss) + '_' + '{:.8f}'.format(val_seed_loss)))
    if is_lowest:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_seg_model.pth' + '{:.4f}'.format(best_seg) + '_' + '{:.4f}'.format(train_loss)))


nan_state = 0
for epoch in range(start_epoch, args['n_epochs']):
    is_best, is_lowest, is_seed_lowest, is_val_lowest = False, False, False, False
    print('Starting epoch {}'.format(epoch))
    if epoch > start_epoch:
        scheduler.step()
    # else:
    #     val_loss, val_iou, val_seed_loss, val_loss = val(epoch)
    #     print('===> val loss: {:.4f}, val iou: {:.4f}, val seed: {:.8f}'.format(val_loss, val_iou, val_seed_loss))
    train_loss, seed_loss = train(epoch)

    seg_iou, val_iou, val_seed_loss, val_loss = val(epoch)

    print('===> train loss: {:.4f}, ===> seed loss: {:.8f}'.format(train_loss, seed_loss))
    print('===> seg iou: {:.4f}, val iou: {:.4f}, val seed: {:.8f}, val loss: {:.8f}'.format(seg_iou, val_iou, val_seed_loss, val_loss))

    logger.add('train', train_loss)
    logger.add('val', seg_iou)
    logger.add('iou', val_iou)
    logger.plot(save=args['save'], save_dir=args['save_dir'])

    is_best = val_iou > best_iou
    best_iou = max(val_iou, best_iou)

    is_lowest = seg_iou > best_seg
    best_seg = max(seg_iou, best_seg)

    if val_seed_loss > 1e-10:
        is_seed_lowest = val_seed_loss < best_seed
        best_seed = min(val_seed_loss, best_seed)
    if val_loss > 1e-10:
        is_val_lowest = val_loss < best_val
        best_val = min(val_loss, best_val)

    if args['save']:
        state = {
            'epoch': epoch,
            'best_iou': best_iou,
            'best_seed': best_seed,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'logger_data': logger.data
        }
        save_checkpoint(state, is_best, val_iou, best_seg, val_seed_loss, train_loss, val_loss, is_val_lowest=is_val_lowest, is_lowest=is_lowest, is_seed_lowest=is_seed_lowest)
