import time

import torch
import torch.nn as nn
from torch_points_kernels import knn


class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

        self.device = device

    def forward(self, coords, features, knn_output):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbors: tuple

            Returns
            -------
            torch.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx, dist = knn_output
        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx) # shape (B, 3, N, K)
        # if USE_CUDA:
        #     neighbors = neighbors.cuda()

        # relative point position encoding
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3).to(features.device)
        return torch.cat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)



class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)

        return self.mlp(features)


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors, device)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors, device)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        knn_output = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.num_neighbors)

        x = self.mlp1(features)

        x = self.lse1(coords, x, knn_output)
        x = self.pool1(x)

        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))


class RandLANet(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=3, decimation=2, device=torch.device('cpu')):
        super(RandLANet, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 32, **decoder_kwargs),
            SharedMLP(64, 8, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, num_classes)
        )
        # self.device = device

        # self = self.to(device)

    def forward(self, input):
        r"""
            Forward pass

            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points

            Returns
            -------
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        input = input.permute(0,2,1)
        N = input.size(1)
        d = self.decimation

        coords = input[...,:3].clone().cpu()
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        coords = coords[:,permutation]
        x = x[:,:,permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:,:N//decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:,:,:N//decimation_ratio]


        # # >>>>>>>>>> ENCODER

        x = self.mlp(x)

        # <<<<<<<<<< DECODER
        for mlp in self.decoder:
            neighbors, _ = knn(
                coords[:,:N//decimation_ratio].cpu().contiguous(), # original set
                coords[:,:d*N//decimation_ratio].cpu().contiguous(), # upsampled set
                1
            ) # shape (B, N, 1)
            neighbors = neighbors.to(input.device)

            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            x_neighbors = torch.gather(x, -2, extended_neighbors)

            x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

            x = mlp(x)

            decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        x = x[:,:,torch.argsort(permutation)]

        scores = self.fc_end(x)

        return scores.squeeze(-1)


class RandLANetWoDec(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=3, decimation=1, device=torch.device('cpu')):
        super(RandLANetWoDec, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ])

        self.mlp = SharedMLP(512, num_classes, activation_fn=nn.ReLU())

    def forward(self, input, with_weight=False):
        r"""
            Forward pass

            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points

            Returns
            -------
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        input = input.permute(0,2,1)
        N = input.size(1)
        d = self.decimation

        coords = input[...,:3].clone().cpu()
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        if not with_weight:
            permutation = torch.randperm(N)
            coords = coords[:,permutation]
            x = x[:,:,permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:,:N//decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:,:,:N//decimation_ratio]


        # # >>>>>>>>>> ENCODER
        x = self.mlp(x)
        if with_weight:
            return x.squeeze(-1), torch.max(x.squeeze(-1), dim=2)[1]
        return x.squeeze(-1)


class RandLANetWeighting(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=3, decimation=1, num_points=1000, device=torch.device('cpu')):
        super(RandLANetWeighting, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ])

        self.mlp_ap = SharedMLP(512, num_classes, activation_fn=nn.ReLU())
        self.mlp_mp = SharedMLP(512, num_classes, activation_fn=nn.ReLU())
        self.ap1 = torch.nn.AvgPool1d(num_points//(decimation*decimation*decimation*decimation))
        self.mp2 = torch.nn.MaxPool1d(num_points//(decimation*decimation*decimation*decimation))
        self.conv_weight = torch.nn.Conv1d(num_classes*2, 1, 1)

    def forward(self, input):
        r"""
            Forward pass

            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points

            Returns
            -------
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        input = input.permute(0,2,1)
        N = input.size(1)
        d = self.decimation

        coords = input[...,:3].clone().cpu()
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        coords = coords[:,permutation]
        x = x[:,:,permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:,:N//decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:,:,:N//decimation_ratio]

        # # >>>>>>>>>> ENCODER

        x1 = self.mlp_ap(x).squeeze(-1)
        x2 = self.mlp_mp(x).squeeze(-1)
        ap_x1 = self.ap1(x1).squeeze(-1)  # B,64
        mp_x2 = self.mp2(x2).squeeze(-1)  # B,64

        weightFeat = self.conv_weight(torch.cat([x1, x2], dim=1))  # B,1,N
        weight = torch.nn.Softmax(2)(weightFeat)
        x = x.squeeze(-1)
        weight_x3 = (weight.expand_as(x) * x).sum(2)

        return torch.cat([ap_x1, mp_x2, weight_x3], dim=1)


class RandLANetWeighting3(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=3, decimation=1, num_points=1000, device=torch.device('cpu')):
        super(RandLANetWeighting3, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ])

        self.mlp_ap = SharedMLP(num_classes, num_classes, activation_fn=nn.ReLU())
        self.mlp_mp = SharedMLP(num_classes, num_classes, activation_fn=nn.ReLU())
        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.mp2 = torch.nn.MaxPool1d(num_points)
        self.conv_weight = torch.nn.Conv1d(num_classes, 1, 1)

        # decoding layers
        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 32, **decoder_kwargs),
            SharedMLP(64, 8, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, num_classes)
        )

    def forward(self, input):
        r"""
            Forward pass

            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points

            Returns
            -------
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        input = input.permute(0,2,1)
        N = input.size(1)
        d = self.decimation

        coords = input[...,:3].clone().cpu()
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        coords = coords[:,permutation]
        x = x[:,:,permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:,:N//decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:,:,:N//decimation_ratio]

        # # >>>>>>>>>> ENCODER
        x = self.mlp(x)

        # <<<<<<<<<< DECODER
        for mlp in self.decoder:
            neighbors, _ = knn(
                coords[:, :N // decimation_ratio].cpu().contiguous(),  # original set
                coords[:, :d * N // decimation_ratio].cpu().contiguous(),  # upsampled set
                1
            )  # shape (B, N, 1)
            neighbors = neighbors.to(input.device)

            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            x_neighbors = torch.gather(x, -2, extended_neighbors)

            x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

            x = mlp(x)

            decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        x = x[:, :, torch.argsort(permutation)]

        x = self.fc_end(x)

        x1 = self.mlp_ap(x).squeeze(-1)
        ap_x1 = self.ap1(x1).squeeze(-1)  # B,64

        weightFeat = self.conv_weight(x1)  # B,1,N
        weight = torch.nn.Softmax(2)(weightFeat)
        x = x.squeeze(-1)
        weight_x3 = (weight.expand_as(x) * x).sum(2)

        return torch.cat([ap_x1, weight_x3], dim=1)


class RandLANetWeighting2(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=3, decimation=1, num_points=1000, device=torch.device('cpu')):
        super(RandLANetWeighting2, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = 1

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64//decimation, num_neighbors, device),
            LocalFeatureAggregation(128//decimation, 64//decimation, num_neighbors, device),
            LocalFeatureAggregation(128//decimation, 64//decimation, num_neighbors, device)
        ])

        self.mlp_ap = SharedMLP(128//decimation, num_classes, activation_fn=nn.ReLU())
        self.mlp_mp = SharedMLP(128//decimation, num_classes, activation_fn=nn.ReLU())
        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.mp2 = torch.nn.MaxPool1d(num_points)
        self.conv_weight = torch.nn.Conv1d(num_classes*2, 1, 1)

    def forward(self, input):
        r"""
            Forward pass

            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points

            Returns
            -------
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        input = input.permute(0,2,1)
        N = input.size(1)
        d = self.decimation

        coords = input[...,:3].clone().cpu()
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        coords = coords[:,permutation]
        x = x[:,:,permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:,:N//decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:,:,:N//decimation_ratio]

        # # >>>>>>>>>> ENCODER

        x1 = self.mlp_ap(x).squeeze(-1)
        x2 = self.mlp_mp(x).squeeze(-1)
        ap_x1 = self.ap1(x1).squeeze(-1)  # B,64
        mp_x2 = self.mp2(x2).squeeze(-1)  # B,64

        weightFeat = self.conv_weight(torch.cat([x1, x2], dim=1))  # B,1,N
        weight = torch.nn.Softmax(2)(weightFeat)
        x = x.squeeze(-1)
        weight_x3 = (weight.expand_as(x) * x).sum(2)

        return torch.cat([ap_x1, mp_x2, weight_x3], dim=1)


class RandLANetWeighting4(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=3, decimation=1, num_points=1000, device=torch.device('cpu')):
        super(RandLANetWeighting4, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = 1

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 32, num_neighbors, device),
            LocalFeatureAggregation(64, 64//decimation, num_neighbors, device),
        ])

        self.mlp_ap = SharedMLP(128 // decimation, num_classes, activation_fn=nn.ReLU())
        self.mlp_mp = SharedMLP(128 // decimation, num_classes, activation_fn=nn.ReLU())
        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.mp2 = torch.nn.MaxPool1d(num_points)
        self.conv_weight = torch.nn.Conv1d(num_classes * 2, 1, 1)

    def forward(self, input, with_weight=False):
        input = input.permute(0,2,1)
        N = input.size(1)
        d = self.decimation

        coords = input[...,:3].clone().cpu()
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        if not with_weight:
            permutation = torch.randperm(N)
            coords = coords[:, permutation]
            x = x[:, :, permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:,:N//decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:,:,:N//decimation_ratio]

        # # >>>>>>>>>> ENCODER

        x1 = self.mlp_ap(x).squeeze(-1)
        x2 = self.mlp_mp(x).squeeze(-1)
        ap_x1 = self.ap1(x1).squeeze(-1)  # B,64
        mp_x2 = self.mp2(x2).squeeze(-1)  # B,64

        weightFeat = self.conv_weight(torch.cat([x1, x2], dim=1))  # B,1,N
        weight = torch.nn.Softmax(2)(weightFeat)
        x = x.squeeze(-1)
        weight_x3 = (weight.expand_as(x) * x).sum(2)
        if with_weight:
            return torch.cat([ap_x1, mp_x2, weight_x3], dim=1), weight

        return torch.cat([ap_x1, mp_x2, weight_x3], dim=1)


class RandLANetWeighting5(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=3, decimation=1, num_points=1000, device=torch.device('cpu')):
        super(RandLANetWeighting5, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = 1

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 32, num_neighbors, device),
            LocalFeatureAggregation(64, 32//decimation, num_neighbors, device),
            LocalFeatureAggregation(64//decimation, 32//decimation, num_neighbors, device),
            LocalFeatureAggregation(64//decimation, 32//decimation, num_neighbors, device),
        ])

        self.mlp_ap = SharedMLP(64 // decimation, num_classes, activation_fn=nn.ReLU())
        self.mlp_mp = SharedMLP(64 // decimation, num_classes, activation_fn=nn.ReLU())
        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.mp2 = torch.nn.MaxPool1d(num_points)
        self.conv_weight = torch.nn.Conv1d(num_classes * 2, 1, 1)

    def forward(self, input):
        input = input.permute(0,2,1)
        N = input.size(1)
        d = self.decimation

        coords = input[...,:3].clone().cpu()
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        coords = coords[:,permutation]
        x = x[:,:,permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:,:N//decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:,:,:N//decimation_ratio]

        # # >>>>>>>>>> ENCODER

        x1 = self.mlp_ap(x).squeeze(-1)
        x2 = self.mlp_mp(x).squeeze(-1)
        ap_x1 = self.ap1(x1).squeeze(-1)  # B,64
        mp_x2 = self.mp2(x2).squeeze(-1)  # B,64

        weightFeat = self.conv_weight(torch.cat([x1, x2], dim=1))  # B,1,N
        weight = torch.nn.Softmax(2)(weightFeat)
        x = x.squeeze(-1)
        weight_x3 = (weight.expand_as(x) * x).sum(2)

        return torch.cat([ap_x1, mp_x2, weight_x3], dim=1)


if __name__ == '__main__':
    import time
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    d_in = 7
    cloud = 1000*torch.randn(1, 1000, d_in).to(device)
    model = RandLANetWeighting(d_in, 64).to(device)
    # model.load_state_dict(torch.load('checkpoints/checkpoint_100.pth'))
    model.eval()

    t0 = time.time()
    pred = model(cloud.permute(0,2,1))
    t1 = time.time()
    # print(pred)
    print(t1-t0)
