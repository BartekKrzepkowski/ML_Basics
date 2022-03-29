import torch
import torch.nn as nn


# Yoon Kim model (https://arxiv.org/abs/1408.5882)
class YoonKimModel(nn.Module):
    def __init__(self, x_channel, b1_channel, y_dim, emb_dim, stride_conv, prob_dropout, context_window):
        super(self).__init__()
        self.context_layers = [nn.Sequential(
            nn.Dropout(p=prob_dropout),
            nn.Conv2d(x_channel, b1_channel, stride=stride_conv,
                      kernel_size=(l_filter_sizes, emb_dim)),
            nn.BatchNorm2d(b1_channel),
            nn.ReLU(),
            nn.Flatten(start_dim=2, end_dim=3),
            nn.AdaptiveAvgPool1d(output_size=1)
        ) for l_filter_sizes in context_window
        ]
        self.dropout = nn.Dropout(p=prob_dropout)
        self.linear = nn.Linear(in_features=b1_channel * len(context_window), out_features=y_dim)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        context_tensors = []

        x = x.permute(0, 2, 1)
        x = self.spatial_dropout(x)
        x = x.permute(0, 2, 1)

        x = x.unsqueeze(1)
        for c_layer in self.context_layers:
            context_tensors.append(c_layer(x))

        x = torch.cat(context_tensors, dim=1).squeeze(-1)
        x = self.dropout(x)
        return self.linear(x)


# AllCNN model (https://arxiv.org/abs/1412.6806)
class AllCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 96, kernel_size=3, padding=2),  # 34
                                 nn.BatchNorm2d(96),
                                 nn.ReLU(),
                                 nn.Conv2d(96, 96, kernel_size=3, padding=1),  # 34
                                 nn.BatchNorm2d(96),
                                 nn.ReLU(),
                                 nn.Conv2d(96, 192, kernel_size=3, stride=2),  # 16
                                 nn.BatchNorm2d(192),
                                 nn.ReLU(),
                                 nn.Conv2d(192, 192, kernel_size=3, padding=1),  # 16
                                 nn.BatchNorm2d(192),
                                 nn.ReLU(),
                                 nn.Conv2d(192, 192, kernel_size=3, stride=2),  # 7
                                 nn.BatchNorm2d(192),
                                 nn.ReLU(),
                                 nn.Conv2d(192, 192, kernel_size=3),  # 5
                                 nn.BatchNorm2d(192),
                                 nn.ReLU(),
                                 nn.Conv2d(192, 192, kernel_size=1),  # 5
                                 nn.Conv2d(192, 10, kernel_size=1)  # 5
                                 )
        self.avg = nn.AvgPool2d(kernel_size=5)

    def forward(self, x):
        x = self.net(x)
        x = self.avg(x)
        x = torch.squeeze(x)
        return x






