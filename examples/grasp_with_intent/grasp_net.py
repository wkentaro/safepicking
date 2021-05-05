import torch
import torch.nn.functional as F


class GraspNet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()

        self._model = model

        if self._model == "rgb":
            self.conv0_rgb = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
            )
        elif self._model == "rgbd":
            self.conv0_rgb = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
            )
            self.conv0_depth = torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
            )
        elif self._model == "ins":
            self.conv0_ins = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
            )
        elif self._model == "insd":
            self.conv0_ins = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
            )
            self.conv0_depth = torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
            )
        else:
            raise ValueError

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            # torch.nn.ReLU(),
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            # torch.nn.ReLU(),
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            # torch.nn.ReLU(),
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # torch.nn.ReLU(),
        )
        self.up7 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            # torch.nn.ReLU(),
        )
        self.up8 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.conv8 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            # torch.nn.ReLU(),
        )
        self.up9 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.conv9 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            # torch.nn.ReLU(),
        )
        self.conv10 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)

    def forward(self, rgb, depth, ins):
        if self._model == "rgb":
            conv0 = self.conv0_rgb(rgb)
        elif self._model == "rgbd":
            conv0_rgb = self.conv0_rgb(rgb)
            conv0_depth = self.conv0_depth(depth)
            conv0 = torch.cat([conv0_rgb, conv0_depth], dim=1)
        elif self._model == "ins":
            conv0 = self.conv0_ins(ins)
        elif self._model == "insd":
            conv0_ins = self.conv0_ins(ins)
            conv0_depth = self.conv0_depth(depth)
            conv0 = torch.cat([conv0_ins, conv0_depth], dim=1)
        else:
            raise ValueError

        conv1 = self.conv1(conv0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)

        up7 = self.up7(
            F.interpolate(
                conv4,
                size=conv3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(
            F.interpolate(
                conv7,
                size=conv2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(
            F.interpolate(
                conv8,
                size=conv1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = self.conv9(merge9)

        conv10 = self.conv10(conv9)

        return conv10
