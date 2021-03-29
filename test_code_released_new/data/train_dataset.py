import torch.utils.data
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import random


class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, img_path_file) -> None:
        super().__init__()
        self.opt = opt
        self.img_pathes = open(img_path_file).readlines()

    def get_au_from_imgpath(self, img_path: str) -> torch.Tensor:
        idx = int(img_path.split("/")[2].split(".")[0])
        csv_data = pd.read_csv("csv/"+img_path.split("/")
                               [1]+".csv")[idx:idx+1]
        au_and_pose_idx = list(range(296, 299)) + list(range(679, 696))
        au_and_pose = csv_data[csv_data.columns[au_and_pose_idx]]
        f = 0
        for i in range(0, len(au_and_pose)):
            au_and_pose_tensor = torch.tensor(
                au_and_pose.values[i]).view(1, -1)
            au_and_pose_tensor[0, 0:3] = (
                au_and_pose_tensor[0, 0:3]-(-0.70))/1.4
            au_and_pose_tensor[0, 3:20] = au_and_pose_tensor[0, 3:20]/5
            purpose_au_and_pose = None
            if f == 0:
                purpose_au_and_pose = au_and_pose_tensor
            else:
                purpose_au_and_pose = torch.cat(
                    [purpose_au_and_pose, au_and_pose_tensor], dim=0)
            f = f+1
        return purpose_au_and_pose

    def get_image_from_path(self, img_path) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB').resize(
            (self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        img_tensor = transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(img))
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(img_tensor.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            img_tensor = img_tensor.index_select(2, idx)
        return img_tensor

    def __getitem__(self, index: int) -> dict:
        input_img_path, acc_img_path = self.img_pathes[index].split(":")
        source_AU, target_AU = self.get_au_from_imgpath(
            input_img_path), self.get_au_from_imgpath(acc_img_path)
        input_img_path, acc_img_path = "img/"+input_img_path.split("/", 1)[1].replace(
            "\n", ""), "img/"+acc_img_path.split("/", 1)[1].replace("\n", "")
        input_img, acc_img = self.get_image_from_path(
            input_img_path), self.get_image_from_path(acc_img_path)
        return {'source': input_img, 'target': acc_img, 'AU_target': target_AU}

    def __len__(self) -> int:
        return len(self.img_pathes)
