'''
ccpnet
'''

import torch
import torch.nn as nn
import cv2
import subprocess
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.utils import ClusterSeedClsWithFilter
from criterions.my_loss import *
from file_utils import *
from config import *
from models.DDETR.misc import NestedTensor, inverse_sigmoid
from utils.mots_util import *
from tqdm import tqdm


class trackFeatExtractor (nn.Module):
    def __init__(self, track_dim):
        super().__init__()
        print('trackFeatExtractor with sigmoid')
        self.track_dim = track_dim
        mid_dim = 64
        from models.erfnetGroupNorm import non_bottleneck_1d_gn, non_bottleneck_1d
        self.lay1 = nn.Sequential(
            non_bottleneck_1d_gn(track_dim, 0.1, 1),
            nn.GroupNorm(8, track_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(track_dim, mid_dim, 2, stride=2, padding=0, output_padding=0, bias=True)
        )
        self.lay2 = nn.Sequential(
            non_bottleneck_1d_gn(mid_dim+64, 0.1, 1),
            nn.Conv2d(mid_dim+64, 32, 1),
        )

    def forward(self, x, low_feat):
        return self.lay2(torch.cat([self.lay1(x), low_feat], dim=1))


class PointTrackLightning(pl.LightningModule):
    SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
    SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
    TIMESTEPS_PER_SEQ = {"0000": 154, "0001": 447, "0002": 233, "0003": 144, "0004": 314, "0005": 297, "0006": 270,
                         "0007": 800, "0008": 390, "0009": 803, "0010": 294, "0011": 373, "0012": 78, "0013": 340,
                         "0014": 106, "0015": 376, "0016": 209, "0017": 145, "0018": 339, "0019": 1059, "0020": 837}
    SEQ_IDS_TEST = ["%04d" % idx for idx in range(29)]
    TIMESTEPS_PER_SEQ_TEST = {'0000': 465, '0015': 701, '0017': 305, '0003': 257, '0001': 147, '0018': 180, '0005': 809,
                              '0022': 436, '0021': 203, '0023': 430, '0012': 694, '0008': 165, '0009': 349, '0020': 173,
                              '0016': 510, '0013': 152, '0004': 421, '0028': 175, '0024': 316, '0019': 404, '0026': 170,
                              '0007': 215, '0014': 850, '0025': 176, '0027': 85, '0011': 774, '0010': 1176, '0006': 114,
                              '0002': 243}
    def __init__(self, num_classes, margin=0.2, n_frames=2, v2=False):
        super().__init__()
        self.args, self.criterion, self.milestones = None, None, None
        print('Creating PointTrack_CCPNet_track with {} franes'.format(n_frames))

        self.num_classes = num_classes
        self.margin = margin
        self.v2 = v2

        from models.erfnetGroupNorm import Encoder0126CCP, Decoder0118, non_bottleneck_1d_gn
        self.backbone = Encoder0126CCP()
        self.decoders = nn.ModuleList()
        mid_dim, pos_dim = 576, 32
        for n in num_classes:
            self.decoders.append(Decoder0118(n, ic=mid_dim, c2=64))

        from models.DDETR.position_encoding import PositionEmbeddingSine
        self.position_encoder = PositionEmbeddingSine(num_pos_feats=pos_dim)

        if not v2:
            track_dim = mid_dim + 64 + pos_dim * 2
            self.trackFeat = nn.Sequential(
                non_bottleneck_1d_gn(track_dim, 0.1, 1),
                nn.GroupNorm(8, track_dim),
                nn.LeakyReLU(0.1, inplace=True),
                non_bottleneck_1d_gn(track_dim, 0.1, 1),
                nn.Conv2d(track_dim, 32, 1),
            )
        else:
            track_dim = mid_dim + pos_dim * 2
            self.trackFeat = trackFeatExtractor(track_dim)
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.cluster = ClusterSeedClsWithFilter()
        self.val_loss, self.val_acc = 100, 0
        self.track_weight = 0.1
        self.n_frames = n_frames
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_output(self, args):
        n_sigma = args['loss_opts']['n_sigma']
        self.args = args
        if 'milestones' in self.args.keys():
            self.milestones = args['milestones']
        self.criterion = eval(self.args['loss_type'])(**self.args['loss_opts'])
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ', output_conv.weight.size())
            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)
            output_conv.weight[:, 2:2+n_sigma, :, :].fill_(0)
            output_conv.bias[2:2+n_sigma].fill_(1)
        # self.load_from('./mots_car_pointtrack_rec/best-val_acc=87.1100.ckpt_V8_rec')

    def load_from(self, path, remove_list=None):
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)['state_dict']
        if remove_list is not None:
            ckpt = remove_key_word(ckpt, remove_list)
        self.load_state_dict(ckpt, strict=False)
        print('resume checkpoint with strict False from %s' % path)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.trackFeat.parameters(), lr=self.args['lr'])
        optimizer = torch.optim.Adam(list(self.trackFeat.parameters())+list(self.decoders.parameters()), lr=self.args['lr'])

        if self.milestones is None:
            def lambda_(epoch):
                return pow((1 - ((epoch) / self.args['n_epochs'])), 0.9)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_,)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.milestones, gamma=0.1)
        return [optimizer], [scheduler]

    def compute_triplet_loss(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        loss = torch.zeros([1]).cuda()
        if mask.float().unique().shape[0] > 1:
            dist_ap, dist_an = [], []
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            dist_ap = torch.cat(dist_ap)
            dist_an = torch.cat(dist_an)
            # Compute ranking hinge loss
            y = torch.ones_like(dist_an)
            loss = self.ranking_loss(dist_an, dist_ap, y).unsqueeze(0)

        return loss

    def forward(self, ims, seg=False):
        if self.v2:
            with torch.no_grad():
                low_feat, output2, output3, output4 = self.backbone(ims, low_level=True)
                output = torch.cat([output2, F.interpolate(output3, scale_factor=(2.0, 2.0), mode='nearest'),
                                    F.interpolate(output4, scale_factor=(4.0, 4.0), mode='nearest')], dim=1)
                inter_output = output.clone()

                b, _, h, w = output.shape
                mask = torch.zeros((1, h, w)).type_as(output).to(torch.bool)
                x = NestedTensor(output, mask)
                posEmbed = self.position_encoder(x).to(x.tensors.dtype)
            embeddings = self.trackFeat(torch.cat([output, posEmbed], dim=1), low_feat)
            embeddings = F.interpolate(embeddings, scale_factor=(2.0, 2.0), mode='nearest')
        else:
            with torch.no_grad():
                low_feat, output2, output3, output4 = self.backbone(ims, low_level=True)
                output = torch.cat([F.interpolate(low_feat, scale_factor=(0.5, 0.5), mode='nearest'), output2,
                                    F.interpolate(output3, scale_factor=(2.0, 2.0), mode='nearest'),
                                    F.interpolate(output4, scale_factor=(4.0, 4.0), mode='nearest')], dim=1)
                inter_output = torch.cat([output2, F.interpolate(output3, scale_factor=(2.0, 2.0), mode='nearest'),
                                          F.interpolate(output4, scale_factor=(4.0, 4.0), mode='nearest')], dim=1)

                b, _, h, w = output.shape
                mask = torch.zeros((1, h, w)).type_as(output).to(torch.bool)
                x = NestedTensor(output, mask)
                posEmbed = self.position_encoder(x).to(x.tensors.dtype)
            embeddings = self.trackFeat(torch.cat([output, posEmbed], dim=1))
            embeddings = F.interpolate(embeddings, scale_factor=(4.0, 4.0), mode='nearest')

        output = torch.cat([decoder.forward(inter_output) for decoder in self.decoders], 1)
        return output, embeddings

    def construct_targets(self, embeddings, instances):
        feats, labels = [], []
        for i in range(len(instances)):
            embedMap, instMap = embeddings[i].permute(0,2,3,1), instances[i]
            for uid in torch.unique(instMap)[1:]:
                labels.append(uid)
                uFeat = embedMap[instMap==uid] # N*32
                feats.append(uFeat.max(0).values.unsqueeze(0))
        if len(feats)==0:
            return torch.tensor([]), torch.tensor([])
        return torch.cat(feats, dim=0).type_as(embeddings[0]), torch.tensor(labels).type_as(embeddings[0]).long()

    def training_step(self, sample, batch_nb):
        if self.n_frames == 2:
            ims = [sample['image'], sample['image2']]
            instances = [sample['instance'].squeeze(1), sample['instance2'].squeeze(1)]
            class_labels = [sample['label'].squeeze(1), sample['label2'].squeeze(1)]
            embeddings, outputs = [], []
            for el in ims:
                output_, emb_ = self(el)
                outputs.append(output_)
                embeddings.append(emb_)
            feats, labels = self.construct_targets(embeddings, instances)
            track_loss = self.compute_triplet_loss(feats, labels) * self.track_weight

            seed_w = sample['seed_w'][0] if 'seed_w' in sample.keys() else None
            loss0, focal_loss0, _ = self.criterion(outputs[0], ims[0], instances[0], class_labels[0], **self.args['loss_w'], seed_w=seed_w)
            loss1, focal_loss1, _ = self.criterion(outputs[1], ims[1], instances[1], class_labels[1], **self.args['loss_w'], seed_w=seed_w)

            flow_loss, focal_loss = track_loss, (focal_loss0+focal_loss1)/2
            loss = (loss0+loss1)/2 + track_loss
        else:
            ims = [sample['image'], sample['image2'], sample['image3']]
            instances = [sample['instance'].squeeze(1), sample['instance2'].squeeze(1), sample['instance3'].squeeze(1)]
            class_labels = [sample['label'].squeeze(1), sample['label2'].squeeze(1), sample['label3'].squeeze(1)]
            embeddings, outputs = [], []
            for el in ims:
                output_, emb_ = self(el)
                outputs.append(output_)
                embeddings.append(emb_)
            feats, labels = self.construct_targets(embeddings, instances)
            track_loss = self.compute_triplet_loss(feats, labels) * self.track_weight

            seed_w = sample['seed_w'][0] if 'seed_w' in sample.keys() else None
            loss0, focal_loss0, _ = self.criterion(outputs[0], ims[0], instances[0], class_labels[0], **self.args['loss_w'], seed_w=seed_w)
            loss1, focal_loss1, _ = self.criterion(outputs[1], ims[1], instances[1], class_labels[1], **self.args['loss_w'], seed_w=seed_w)
            loss2, focal_loss2, _ = self.criterion(outputs[2], ims[2], instances[2], class_labels[2], **self.args['loss_w'], seed_w=seed_w)
            flow_loss, focal_loss = track_loss, (focal_loss0+focal_loss1+focal_loss2)/2
            loss = (loss0+loss1+loss2)/3 + track_loss
        return {'loss': loss, 'focal_loss': focal_loss, 'flow_loss': flow_loss}

    def training_step_end(self, batch_parts):
        return {'loss': batch_parts['loss'].mean(), 'focal_loss': batch_parts['focal_loss'].mean(), 'flow_loss': batch_parts['flow_loss'].mean()}

    def training_epoch_end(self, train_step_outputs):
        if self.global_rank == 0:
            train_loss, focal_loss, flow_loss = 0, 0, 0
            for v in train_step_outputs:
                train_loss += v['loss'].item()
                focal_loss += v['focal_loss'].item()
                flow_loss += v['flow_loss'].item()
            train_loss = train_loss / len(train_step_outputs)
            focal_loss = focal_loss / len(train_step_outputs)
            flow_loss = flow_loss / len(train_step_outputs)
            print('')
            print('===> train loss: {:.8f}, ===> focal_loss: {:.8f}, ===> flow_loss: {:.8f}'.format(train_loss, focal_loss, flow_loss))
            print('Learning Rate {:.8f}'.format(self.trainer.optimizers[0].param_groups[0]['lr']))

    def validation_epoch_start(self):
        self.eval()
        torch.cuda.empty_cache()

    def construct_targets_val(self, embeddings, instances):
        feats, masks, labels = [], [], []
        for i in range(len(instances)):
            embedMap, instMap = embeddings[i].permute(0,2,3,1), instances[i]
            for uid in torch.unique(instMap)[1:]:
                labels.append(uid)
                mask_ = instMap==uid
                masks.append(mask_.bool())
                uFeat = embedMap[mask_]
                feats.append(uFeat.max(0).values.unsqueeze(0))
        if len(feats)==0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        return torch.cat(feats, dim=0).type_as(embeddings[0]), torch.cat(masks, dim=0), torch.tensor(labels).type_as(embeddings[0]).long()

    def validation_step(self, sample, batch_idx):
        with torch.no_grad():
            ims = sample['image']
            instances = sample['instance'].squeeze(1)
            class_labels = sample['label'].squeeze(1)
            output, embeddings = self(ims, seg=True)
            loss, focal_loss, ious = self.criterion(output, ims, instances, class_labels, **self.args['loss_w'], iou=True)

            sizes = sample['im_shape'].squeeze(1)
            im_names = sample['im_name']
            args = self.args
            if 'mots' in self.args['save_dir']:
                for ind in range(ims.shape[0]):
                    video, frameCount = im_names[ind].split('/')[-2:]
                    frameCount = int(float(frameCount.split('.')[0]))
                    # base = os.path.join(args['seg_dir'], video + '_' + str(frameCount) + '.pkl')
                    (w, h) = sizes[ind]
                    instance_maps = self.cluster(output, threshold=args['threshold'], min_pixel=args['min_pixel'],
                                       min_inst_pixel=args['min_inst_pixel'] if "min_inst_pixel" in args.keys() else args['min_pixel'],
                                       min_seed_thresh=args['min_seed_thresh'] if "min_seed_thresh" in args.keys() else 0.5,
                                       dist_thresh=args['dist_thresh'] if "dist_thresh" in args.keys() else 0.5,
                                       inst_ratio=args['inst_ratio'] if "inst_ratio" in args.keys() else 0.5,
                                       n_sigma=args["n_sigma"] if "n_sigma" in args.keys() else 2,
                                       avg_seed=args["avg_seed"] if "avg_seed" in args.keys() else 0.0)

                    embeddings, instance_maps = embeddings[:, :, :h.item(), :w.item()], instance_maps[:, :h.item(), :w.item()]
                    instEmbeds, instMasks, instUids = self.construct_targets_val([embeddings], [instance_maps])

                    # save embeddings
                    embedPath = video + '_' + str(frameCount) + '.npz'
                    embed_np, uid_np, masks_np = instEmbeds.cpu().numpy(), instUids.cpu().numpy(), instMasks.cpu().numpy()
                    np.savez_compressed(os.path.join(args['seg_dir'], embedPath), embed=embed_np, uid=uid_np, mask=masks_np)
            return {'val_loss': loss, 'focal_loss': focal_loss, 'iou_sum': sum(ious), 'iou_count': len(ious)}

    def validation_step_end(self, batch_parts):
        # do something with both outputs
        iou_sum = sum(batch_parts['iou_sum']) if isinstance(batch_parts['iou_sum'], list) else batch_parts['iou_sum']
        iou_count = sum(batch_parts['iou_count']) if isinstance(batch_parts['iou_count'], list) else batch_parts['iou_count']
        if iou_count>0 and iou_sum / float(iou_count) < 0.7 and iou_count > 5:
            self.validation_epoch_end([])
        return {
            'val_loss': batch_parts['val_loss'].mean(),
            'focal_loss': batch_parts['focal_loss'].mean(),
            'iou_sum': sum(batch_parts['iou_sum']) if isinstance(batch_parts['iou_sum'], list) else batch_parts['iou_sum'],
            'iou_count': sum(batch_parts['iou_count']) if isinstance(batch_parts['iou_count'], list) else batch_parts['iou_count']
        }

    def validation_epoch_end(self, validation_step_outputs):
        if len(validation_step_outputs) < 10: # sanity check
            return
        val_loss, focal_loss, iou_sum, iou_count = 0,0,0,0
        for v in validation_step_outputs:
            val_loss += v['val_loss'].item()
            focal_loss += v['focal_loss'].item()
            iou_sum += v['iou_sum']
            iou_count += v['iou_count']
        val_loss = val_loss/len(validation_step_outputs)
        focal_loss = focal_loss/len(validation_step_outputs)
        val_iou = iou_sum / iou_count

        self.eval()
        torch.cuda.empty_cache()
        if self.global_rank == 0 and 'mots' in self.args['save_dir']:
            # tracking
            self.mots_root = os.path.join(rootDir, self.args['seg_dir'])
            self.mots_car_sequence = []
            ids = self.SEQ_IDS_VAL
            for valF in ids:
                nums = self.TIMESTEPS_PER_SEQ[valF]
                for i in range(nums):
                    npzPath = os.path.join(self.mots_root, valF + '_' + str(i) + '.npz')
                    if os.path.isfile(npzPath):
                        self.mots_car_sequence.append(npzPath)

            # tracker init
            trackHelper = TrackHelper0226(self.args['track_dir'], self.margin, car='car' in self.args['save_dir'])

            # tracking
            for _, npzPath in enumerate(self.mots_car_sequence):
                embedUids = np.load(npzPath)
                embeds, uids, masks = embedUids['embed'], embedUids['uid'], embedUids['mask']
                subf, frameCount = os.path.basename(npzPath)[:-4].split('_')
                frameCount = int(float(frameCount))
                trackHelper.tracking(subf, frameCount, embeds, masks)
            trackHelper.export_last_video()

            p = subprocess.run([pythonPath, "-u", "eval.py",
                                os.path.join(rootDir, self.args['track_dir']), kittiRoot + "instances", "val.seqmap"],
                               stdout=subprocess.PIPE, cwd=os.path.join(rootDir, "datasets/mots_tools/mots_eval"))
            pout = p.stdout.decode("utf-8")
            if 'person' in self.args['save_dir']:
                class_str = "Evaluate class: Pedestrians"
            else:
                class_str = "Evaluate class: Cars"
            pout = pout[pout.find(class_str):]
            print(pout[pout.find('all   '):][6:126].strip())
            acc_str = pout[pout.find('all   '):][6:26].strip().split(' ')[0]
            ids_str = [el for el in pout[pout.find('all   '):][6:126].strip().split(' ') if len(el) > 0][16]
            val_loss = float(ids_str)
            if len(acc_str) > 0:
                val_acc = float(acc_str)
            else:
                val_acc = 0.0
            print('####### val_acc: {:.4f}, seg iou: {:.4f}, val seed: {:.8f}, val loss: {:.8f}'.format(val_acc, val_iou, focal_loss, val_loss))
            self.log('seg_iou', val_iou)
            self.log('val_acc', val_acc)
            self.log('val_loss', val_loss)
        else:
            val_acc, val_iou = 0.0, 0.0

    def test_step(self, sample, batch_idx):
        with torch.no_grad():
            ims = sample['image']
            instances = sample['instance'].squeeze(1)
            class_labels = sample['label'].squeeze(1)
            output, embeddings = self(ims, seg=True)
            loss, focal_loss, ious = self.criterion(output, ims, instances, class_labels, **self.args['loss_w'], iou=True)

            sizes = sample['im_shape'].squeeze(1)
            im_names = sample['im_name']
            args = self.args
            if 'mots' in self.args['save_dir']:
                for ind in range(ims.shape[0]):
                    video, frameCount = im_names[ind].split('/')[-2:]
                    frameCount = int(float(frameCount.split('.')[0]))
                    # base = os.path.join(args['seg_dir'], video + '_' + str(frameCount) + '.pkl')
                    (w, h) = sizes[ind]
                    instance_maps = self.cluster(output, threshold=args['threshold'], min_pixel=args['min_pixel'],
                                           min_inst_pixel=args['min_inst_pixel'] if "min_inst_pixel" in args.keys() else args['min_pixel'],
                                           min_seed_thresh=args['min_seed_thresh'] if "min_seed_thresh" in args.keys() else 0.5,
                                           dist_thresh=args['dist_thresh'] if "dist_thresh" in args.keys() else 0.5,
                                           inst_ratio=args['inst_ratio'] if "inst_ratio" in args.keys() else 0.5,
                                           n_sigma=args["n_sigma"] if "n_sigma" in args.keys() else 2,
                                           avg_seed=args["avg_seed"] if "avg_seed" in args.keys() else 0.0)

                    embeddings, instance_maps = embeddings[:, :, :h.item(), :w.item()], instance_maps[:, :h.item(), :w.item()]
                    instEmbeds, instMasks, instUids = self.construct_targets_val([embeddings], [instance_maps])

                    # save embeddings
                    embedPath = video + '_' + str(frameCount) + '.npz'
                    embed_np, uid_np, masks_np = instEmbeds.cpu().numpy(), instUids.cpu().numpy(), instMasks.cpu().numpy()
                    mkdir_if_no(args['seg_dir'])
                    np.savez_compressed(os.path.join(args['seg_dir'], embedPath), embed=embed_np, uid=uid_np, mask=masks_np)
            return {'val_loss': loss, 'focal_loss': focal_loss, 'iou_sum': sum(ious), 'iou_count': len(ious)}

    def test_step_end(self, batch_parts):
        # do something with both outputs
        return {
            'val_loss': batch_parts['val_loss'].mean(),
            'focal_loss': batch_parts['focal_loss'].mean(),
            'iou_sum': sum(batch_parts['iou_sum']) if isinstance(batch_parts['iou_sum'], list) else batch_parts['iou_sum'],
            'iou_count': sum(batch_parts['iou_count']) if isinstance(batch_parts['iou_count'], list) else batch_parts['iou_count']
        }

    def test_epoch_end(self, validation_step_outputs):
        val_loss, focal_loss, iou_sum, iou_count = 0,0,0,0
        for v in validation_step_outputs:
            val_loss += v['val_loss'].item()
            focal_loss += v['focal_loss'].item()
            iou_sum += v['iou_sum']
            iou_count += v['iou_count']
        val_loss = val_loss/len(validation_step_outputs)
        focal_loss = focal_loss/len(validation_step_outputs)
        val_iou = iou_sum / iou_count

        self.eval()
        torch.cuda.empty_cache()
        if self.global_rank == 0 and 'mots' in self.args['save_dir']:
            # tracking
            self.mots_root = os.path.join(rootDir, self.args['seg_dir'])
            self.mots_car_sequence = []
            ids = self.SEQ_IDS_VAL
            for valF in ids:
                nums = self.TIMESTEPS_PER_SEQ[valF]
                for i in range(nums):
                    npzPath = os.path.join(self.mots_root, valF + '_' + str(i) + '.npz')
                    if os.path.isfile(npzPath):
                        self.mots_car_sequence.append(npzPath)

            # tracker init
            trackHelper = TrackHelper0226(self.args['track_dir'], self.margin, car='car' in self.args['save_dir'])
            # if 'person' in self.args['save_dir']:
            #     trackHelper = TrackHelper(self.args['track_dir'], self.margin, alive_person=30, car=False, MOTS20=False,
            #                               mask_iou=True)
            # else:
            #     trackHelper = TrackHelper(self.args['track_dir'], self.margin, alive_car=30, car=True, MOTS20=False,
            #                               mask_iou=True)

            # tracking
            for _, npzPath in enumerate(self.mots_car_sequence):
                embedUids = np.load(npzPath)
                embeds, uids, masks = embedUids['embed'], embedUids['uid'], embedUids['mask']
                subf, frameCount = os.path.basename(npzPath)[:-4].split('_')
                frameCount = int(float(frameCount))
                trackHelper.tracking(subf, frameCount, embeds, masks)
            trackHelper.export_last_video()

            p = subprocess.run([pythonPath, "-u", "eval.py",
                                os.path.join(rootDir, self.args['track_dir']), kittiRoot + "instances", "val.seqmap"],
                               stdout=subprocess.PIPE, cwd=os.path.join(rootDir, "datasets/mots_tools/mots_eval"))
            pout = p.stdout.decode("utf-8")
            if 'person' in self.args['save_dir']:
                class_str = "Evaluate class: Pedestrians"
            else:
                class_str = "Evaluate class: Cars"
            pout = pout[pout.find(class_str):]
            print(pout[pout.find('all   '):][6:126].strip())
            acc_str = pout[pout.find('all   '):][6:26].strip().split(' ')[0]
            ids_str = [el for el in pout[pout.find('all   '):][6:126].strip().split(' ') if len(el) > 0][16]
            val_loss = float(ids_str)
            if len(acc_str) > 0:
                val_acc = float(acc_str)
            else:
                val_acc = 0.0
            print('####### val_acc: {:.4f}, seg iou: {:.4f}, val seed: {:.8f}, val loss: {:.8f}'.format(val_acc, val_iou, focal_loss, val_loss))
