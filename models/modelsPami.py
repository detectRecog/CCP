"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BranchedERFNet import LocationEmbedding
from models.randLA import RandLANet, RandLANetWoDec, RandLANetWeighting, RandLANetWeighting2, RandLANetWeighting3, RandLANetWeighting4, RandLANetWeighting5
from models.relation import RelationModule


class PoseNetFeatOffsetEmbRandLA(nn.Module):
    # bn with border
    def __init__(self, num_points=1000, ic=7, border_points=500, border_ic=6, output_dim=64, category=False, FG=False, ENV=False, POS=False):
        super(PoseNetFeatOffsetEmbRandLA, self).__init__()
        self.category = category
        bc = 256
        self.borderConv = RandLANetWoDec(9, 256)
        self.FGConv = RandLANetWeighting(6, 64)
        self.borderPool = torch.nn.AvgPool1d(border_points)
        # self.FGPool = torch.nn.MaxPool1d(num_points)
        self.FGPool = torch.nn.AvgPool1d(num_points)

        self.conv_weight = torch.nn.Conv1d(128, 1, 1)

        self.last_emb = nn.Sequential(
            nn.Linear(3 * output_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim * 3)
        )
        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.mp2 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points

        # FG
        self.FG = FG
        self.last_emb_FG = nn.Sequential(
            # nn.Dropout(0.01),
            nn.Linear(640, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_dim)
        )
        # ENV
        self.ENV = ENV
        self.last_emb_ENV = nn.Sequential(
            # nn.Dropout(0.01),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_dim)
        )
        # POS
        self.POS = POS
        self.last_emb_POS = nn.Sequential(
            # nn.Dropout(0.01),
            nn.Linear(64, 64),
            # nn.Linear(112, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, inp, borders, spatialEmbs, with_weight=False):
        if self.POS:
            return self.last_emb_POS(torch.cat([spatialEmbs], dim=1))
        elif self.ENV:
            # append zero to complement 3-dim XYZ pointcloud
            env_inp = torch.cat([borders[:, 3:5], torch.zeros(borders[:, 3:4].shape).type_as(borders), borders[:, :3], borders[:, 5:]], dim=1)
            # env_features = self.borderConv(env_inp)
            env_features = self.borderPool(self.borderConv(env_inp)).squeeze(-1)
            return self.last_emb_ENV(env_features)
        elif self.FG:
            # x, emb = inp[:, -2:], inp[:, :-2]
            fg_inp = torch.cat([inp[:, -2:], torch.zeros(inp[:, -2:-1].shape).type_as(inp), inp[:, :-2]], dim=1)
            fg_features = self.FGConv(fg_inp)
            return self.last_emb_FG(fg_features)
        else:
            with torch.no_grad():
                env_inp = torch.cat([borders[:, 3:5], torch.zeros(borders[:, 3:4].shape).type_as(borders), borders[:, :3], borders[:, 5:]], dim=1)
                env_features = self.borderPool(self.borderConv(env_inp)).squeeze(-1)
                fg_inp = torch.cat([inp[:, -2:], torch.zeros(inp[:, -2:-1].shape).type_as(inp), inp[:, :-2]], dim=1)
                fg_features = self.FGConv(fg_inp)

            FG_emb = self.last_emb_FG(fg_features)
            POS_emb = self.last_emb_POS(torch.cat([spatialEmbs], dim=1))
            ENV_emb = self.last_emb_ENV(env_features)

            outp = torch.cat([FG_emb, POS_emb, ENV_emb], dim=1)
            outp = torch.cat([self.last_emb(outp), FG_emb, POS_emb, ENV_emb], dim=1)

            return outp


class PoseNetFeatOffsetEmbRandLARelation(nn.Module):
    # bn with border
    def __init__(self, num_points=1000, ic=7, border_points=500, border_ic=6, output_dim=64, category=False, FG=False, ENV=False, POS=False):
        super(PoseNetFeatOffsetEmbRandLARelation, self).__init__()
        self.category = category
        bc = 256
        self.borderConv = RandLANetWoDec(9, 256)
        self.FGConv = RandLANetWeighting4(6, 64, num_neighbors=4, decimation=1)
        # self.FGConv = RandLANetWoDec(6, 640)
        # self.borderPool = torch.nn.MaxPool1d(border_points)
        self.borderPool = torch.nn.AvgPool1d(border_points)
        # self.FGPool = torch.nn.MaxPool1d(num_points)
        self.FGPool = torch.nn.AvgPool1d(num_points)

        self.conv_weight = torch.nn.Conv1d(128, 1, 1)

        self.last_emb = nn.Sequential(
            nn.Linear(3 * output_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim * 3)
        )
        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.mp2 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points

        # FG
        self.FG = FG
        self.last_emb_FG = nn.Sequential(
            # nn.Dropout(0.01),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_dim)
        )
        # ENV
        self.ENV = ENV
        self.last_emb_ENV = nn.Sequential(
            # nn.Dropout(0.01),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_dim)
        )
        # POS
        self.POS = POS
        self.last_emb_POS = nn.Sequential(
            # nn.Dropout(0.01),
            nn.Linear(64, 64),
            # nn.Linear(112, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_dim)
        )
        n_relation = 2
        self.relation = nn.ModuleList([RelationModule(n_relations=n_relation, key_feature_dim=output_dim // n_relation,
                                                      appearance_feature_dim=output_dim,
                                                      geo_feature_dim=1)])
        self.weight = nn.Linear(3*output_dim, 9)

    def forward(self, inp, borders, spatialEmbs, with_weight=False):
        if self.POS:
            return self.last_emb_POS(torch.cat([spatialEmbs], dim=1))
        elif self.ENV:
            # append zero to complement 3-dim XYZ pointcloud
            env_inp = torch.cat([borders[:, 3:5], torch.zeros(borders[:, 3:4].shape).type_as(borders), borders[:, :3], borders[:, 5:]], dim=1)
            # env_features = self.borderConv(env_inp)
            env_features = self.borderPool(self.borderConv(env_inp)).squeeze(-1)
            return self.last_emb_ENV(env_features)
        elif self.FG:
            # x, emb = inp[:, -2:], inp[:, :-2]
            fg_inp = torch.cat([inp[:, -2:], torch.zeros(inp[:, -2:-1].shape).type_as(inp), inp[:, :-2]], dim=1)
            fg_features = self.FGConv(fg_inp)
            return self.last_emb_FG(fg_features)
        else:
            if with_weight:
                fg_inp = torch.cat([inp[:, -2:], torch.zeros(inp[:, -2:-1].shape).type_as(inp), inp[:, :-2]], dim=1)
                fg_features, weight = self.FGConv(fg_inp, with_weight=True)
                env_inp = torch.cat([borders[:, 3:5], torch.zeros(borders[:, 3:4].shape).type_as(borders), borders[:, :3], borders[:, 5:]], dim=1)
                env_feats, bg_inds = self.borderConv(env_inp, with_weight=True)
                env_features = self.borderPool(env_feats).squeeze(-1)

                FG_emb = self.last_emb_FG(fg_features)
                POS_emb = self.last_emb_POS(torch.cat([spatialEmbs], dim=1))
                ENV_emb = self.last_emb_ENV(env_features)

                embs = torch.cat([FG_emb.unsqueeze(1), ENV_emb.unsqueeze(1), POS_emb.unsqueeze(1)], dim=1)
                weights = torch.sigmoid(self.weight(torch.cat([FG_emb, ENV_emb, POS_emb], dim=1)))
                dists = weights.view(inp.shape[0], 3, 3)
                embedsRelation = []
                for i in range(inp.shape[0]):
                    embeds, xyxys = embs[i], dists[i]
                    for rr in self.relation:
                        embeds = rr(embeds, xyxys)
                    embedsRelation.append(embeds.view(-1).unsqueeze(0))
                outp = torch.cat(embedsRelation, dim=0)
                return outp, weight, bg_inds
            else:
                with torch.no_grad():
                    env_inp = torch.cat([borders[:, 3:5], torch.zeros(borders[:, 3:4].shape).type_as(borders), borders[:, :3], borders[:, 5:]], dim=1)
                    env_features = self.borderPool(self.borderConv(env_inp)).squeeze(-1)
                    fg_inp = torch.cat([inp[:, -2:], torch.zeros(inp[:, -2:-1].shape).type_as(inp), inp[:, :-2]], dim=1)
                    fg_features = self.FGConv(fg_inp)

                    FG_emb = self.last_emb_FG(fg_features)
                    POS_emb = self.last_emb_POS(torch.cat([spatialEmbs], dim=1))
                    ENV_emb = self.last_emb_ENV(env_features)

                embs = torch.cat([FG_emb.unsqueeze(1), ENV_emb.unsqueeze(1), POS_emb.unsqueeze(1)], dim=1)
                weights = torch.sigmoid(self.weight(torch.cat([FG_emb, ENV_emb, POS_emb], dim=1)))
                dists = weights.view(inp.shape[0], 3, 3)
                embedsRelation = []
                for i in range(inp.shape[0]):
                    embeds, xyxys = embs[i], dists[i]
                    for rr in self.relation:
                        embeds = rr(embeds, xyxys)
                    embedsRelation.append(embeds.view(-1).unsqueeze(0))
                outp = torch.cat(embedsRelation, dim=0)
                return outp


class TrackerOffsetEmbRandLA(nn.Module):
    # for uv offset and category
    def __init__(self, margin=0.3, num_points=250, border_ic=6, env_points=200, category=False, outputD=64, FG=False, ENV=False, POS=False):
        super().__init__()
        self.point_feat = PoseNetFeatOffsetEmbRandLARelation(num_points=num_points, ic=3, border_points=env_points, border_ic=border_ic, output_dim=outputD, category=True, FG=FG, ENV=ENV, POS=POS)
        # self.point_feat = PoseNetFeatOffsetEmbRandLA(num_points=num_points, ic=3, border_points=env_points, border_ic=border_ic, output_dim=outputD, category=True, FG=FG, ENV=ENV, POS=POS)
        self.num_points = num_points
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.embedding = LocationEmbedding

    def init_output(self, n_sigma=1):
        pass

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

    def forward(self, points, labels, xyxys, infer=False, visualize=False):
        points, xyxys = points[0], xyxys[0]
        embeds = self.embedding(xyxys)
        envs = points[:,self.num_points:]
        points = points[:,:self.num_points, :5]
        if infer:
            return self.inference(points, envs, embeds)
        elif visualize:
            embeds, point_weights, bg_inds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), embeds, with_weight=True)
            return embeds, point_weights, bg_inds
        else:
            embeds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), embeds)
            labels = labels[0]
            return self.compute_triplet_loss(embeds, labels)

    def inference(self, points, envs, embeds):
        # assert points.shape[0] == 1
        embeds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), embeds)
        return embeds


def LocationEmbedding(f_g, dim_g=64, wave_len=1000):
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    position_mat = torch.cat((cx, cy, w, h), -1)

    feat_range = torch.arange(dim_g / 8).cuda()
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, -1)
    position_mat = position_mat.view(f_g.shape[0], 4, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(f_g.shape[0], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)
    return embedding



class PointFeatFuse3P(nn.Module):
    # three path
    def __init__(self, num_points=250, ic=7, oc=64, maxpool=True):
        super(PointFeatFuse3P, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(ic, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.e_conv3 = torch.nn.Conv1d(128, 256, 1)

        self.t_conv1 = torch.nn.Conv1d(3, 64, 1)
        self.t_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.t_conv3 = torch.nn.Conv1d(128, 128, 1)

        self.conv4 = torch.nn.Conv1d(512, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, oc, 1)

        if maxpool:
            self.pool = torch.nn.MaxPool1d(num_points)
        else:
            self.pool = torch.nn.AvgPool1d(num_points)

        self.num_points = num_points

    def forward(self, x, emb, t, withInd=False):
        x = F.leaky_relu(self.conv1(x))
        emb = F.leaky_relu(self.e_conv1(emb))
        t = F.leaky_relu(self.t_conv1(t))

        x = F.leaky_relu(self.conv2(x))
        emb = F.leaky_relu(self.e_conv2(emb))
        t = F.leaky_relu(self.t_conv2(t))

        x = F.leaky_relu(self.conv3(x))
        emb = F.leaky_relu(self.e_conv3(emb))
        t = F.leaky_relu(self.t_conv3(t))

        pointfeat_2 = torch.cat((x, emb, t), dim=1)

        x1 = F.leaky_relu(self.conv4(pointfeat_2))
        x1 = F.leaky_relu(self.conv5(x1))
        x1 = F.leaky_relu(self.conv6(x1))
        if withInd:
            return self.pool(x1).squeeze(-1), torch.max(x1, dim=2)[1]
        return self.pool(x1).squeeze(-1)


class PoseNetFeatOffsetEmbPT(nn.Module):
    # old PointTrack version
    # bn with border
    def __init__(self, num_points=250, ic=7, border_points=200, border_ic=6, output_dim=64, category=False, v23=False):
        super(PoseNetFeatOffsetEmbPT, self).__init__()
        self.category = category
        bc = 256
        self.borderConv = PointFeatFuse3P(ic=border_ic, oc=bc, num_points=border_points)

        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv1_bn = nn.BatchNorm1d(64)
        self.conv2_bn = nn.BatchNorm1d(128)
        self.conv3_bn = nn.BatchNorm1d(256)

        self.e_conv1 = torch.nn.Conv1d(ic, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.e_conv3 = torch.nn.Conv1d(128, 256, 1)
        self.e_conv1_bn = nn.BatchNorm1d(64)
        self.e_conv2_bn = nn.BatchNorm1d(128)
        self.e_conv3_bn = nn.BatchNorm1d(256)

        self.conv4 = torch.nn.Conv1d(512, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 64, 1)
        self.conv4_bn = nn.BatchNorm1d(256)
        self.conv5_bn = nn.BatchNorm1d(512)

        self.conv7 = torch.nn.Conv1d(512, 256, 1)
        self.conv8 = torch.nn.Conv1d(256, 512, 1)
        self.conv9 = torch.nn.Conv1d(512, 64, 1)
        self.conv7_bn = nn.BatchNorm1d(256)
        self.conv8_bn = nn.BatchNorm1d(512)

        self.conv_weight = torch.nn.Conv1d(128, 1, 1)

        self.last_emb = nn.Sequential(
            nn.Linear(704+bc, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )
        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.mp2 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points

    def forward(self, inp, borders, spatialEmbs, with_weight=False):
        x, emb = inp[:,-2:], inp[:,:-2]
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)))
        emb = F.leaky_relu(self.e_conv1_bn(self.e_conv1(emb)))

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)))
        emb = F.leaky_relu(self.e_conv2_bn(self.e_conv2(emb)))

        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))          # B,256,N
        emb = F.leaky_relu(self.e_conv3_bn(self.e_conv3(emb)))  # B,256,N

        pointfeat_2 = torch.cat((x, emb), dim=1)

        x1 = F.leaky_relu(self.conv4_bn(self.conv4(pointfeat_2)))
        x1 = F.leaky_relu(self.conv5_bn(self.conv5(x1)))
        x1 = F.leaky_relu(self.conv6(x1))                       # B,64,N
        ap_x1 = self.ap1(x1).squeeze(-1)                        # B,64

        x2 = F.leaky_relu(self.conv7_bn(self.conv7(pointfeat_2)))
        x2 = F.leaky_relu(self.conv8_bn(self.conv8(x2)))
        x2 = F.leaky_relu(self.conv9(x2))                       # B,64,N
        mp_x2 = self.mp2(x2).squeeze(-1)                        # B,64

        weightFeat = self.conv_weight(torch.cat([x1, x2], dim=1))   #B,1,N
        weight = torch.nn.Softmax(2)(weightFeat)
        weight_x3 = (weight.expand_as(pointfeat_2) * pointfeat_2).sum(2)

        if with_weight:
            border_feat, bg_inds = self.borderConv(borders[:, 3:5], borders[:, :3], borders[:, 5:], withInd=with_weight)
            x = torch.cat([ap_x1, mp_x2, weight_x3, border_feat, spatialEmbs], dim=1)
            outp = self.last_emb(x)
            return outp, weight, bg_inds
        else:
            if self.category:
                border_feat = self.borderConv(borders[:, 3:5], borders[:, :3], borders[:, 5:])
            else:
                border_feat = self.borderConv(borders[:, 3:5], borders[:, :3])

        x = torch.cat([ap_x1, mp_x2, weight_x3, border_feat, spatialEmbs], dim=1)
        outp = self.last_emb(x)
        return outp


class TrackerOffsetEmbPT(nn.Module):
    # for uv offset and category
    def __init__(self, margin=0.3, num_points=250, border_ic=6, env_points=200, category=False, outputD=64, v23=False):
        super().__init__()
        self.point_feat = PoseNetFeatOffsetEmbPT(num_points=num_points, ic=3, border_points=env_points, border_ic=border_ic, output_dim=outputD, category=True)
        self.num_points = num_points
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.embedding = LocationEmbedding

    def init_output(self, n_sigma=1):
        pass

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

    def forward(self, points, labels, xyxys, infer=False, visualize=False):
        points, xyxys = points[0], xyxys[0]
        embeds = self.embedding(xyxys)
        envs = points[:,self.num_points:]
        points = points[:,:self.num_points, :5]
        if infer:
            return self.inference(points, envs, embeds)
        elif visualize:
            embeds, point_weights, bg_inds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), embeds, with_weight=True)
            return embeds, point_weights, bg_inds
        else:
            embeds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), embeds)
            labels = labels[0]
            return self.compute_triplet_loss(embeds, labels)

    def inference(self, points, envs, embeds):
        # assert points.shape[0] == 1
        embeds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), embeds)
        return embeds