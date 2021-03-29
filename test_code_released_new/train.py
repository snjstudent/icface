import os
from random import shuffle

from numpy.core.function_base import add_newdoc
from options.train_options import TrainOptions
from data.train_dataset import Train_Dataset
from models.models import create_model
#from util.visualizer import Visualizer
#from util import html
import pdb
import torch
from torch.utils.tensorboard import SummaryWriter, writer


opt = TrainOptions().parse()
dataset = Train_Dataset(opt, 'train.txt')
dataset_val = Train_Dataset(opt, 'val.txt')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
dataloader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=32, shuffle=True)
model = create_model(opt)
writer = SummaryWriter()
#visualizer = Visualizer(opt)
# create website
#webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test


def add_write_loss(name, loss, count):
    writer.add_scalar("Loss/"+name, loss, count)


def train():
    len_dataset = len(dataloader)
    for epoch in range(5000):
        for i, data in enumerate(dataloader):
            print(
                f'[train] epoch {epoch+1} , {(i/len_dataset)*100}% complited')
            model.set_input(data)
            model.train()
            count = len_dataset*epoch+i
            add_write_loss("Neu/dis", model.loss_G_GANE, count)
            add_write_loss("Neu/recon", model.recon, count)
            add_write_loss("Neu/identity", model.recon_Light, count)
            add_write_loss("Neu/AU", model.sal_loss2, count)
            add_write_loss("Attribute/dis", model.loss_G_GANE_attribute, count)
            add_write_loss("Attribute/recon", model.R, count)
            add_write_loss("Attribute/identity",
                           model.recon_Light_attribute, count)
            add_write_loss("Attribute/AU", model.sal_loss2_attribute, count)
            # add_write_loss("Dis/All", model.loss_discriminator, count)
            # add_write_loss("Dis/Neu", model.loss_G_GANE+model.sal_loss2, count)
            # add_write_loss(
            #     "Dis/Attribute", model.loss_G_GANE_attribute+model.sal_loss2_attribute, count)
            # add_write_loss(
            #     "Gen/Neu", model.loss_GR, count)
            # add_write_loss(
            #     "Gen/Attribute", model.loss_generator_attribute, count)]
        for i, data in enumerate(dataloader_val):
            print(f'[val] epoch {epoch+1} , {(i/len_dataset)*100}% complited')
            model.set_input(data)
            model.validate()
            count = len_dataset*epoch+i
            add_write_loss("Neu/dis_val", model.loss_G_GANE, count)
            add_write_loss("Neu/recon_val", model.recon, count)
            add_write_loss("Neu/identity_val", model.recon_Light, count)
            add_write_loss("Neu/AU_val", model.sal_loss2, count)
            add_write_loss("Attribute/dis_val",
                           model.loss_G_GANE_attribute, count)
            add_write_loss("Attribute/recon_val", model.R, count)
            add_write_loss("Attribute/identity_val",
                           model.recon_Light_attribute, count)
            add_write_loss("Attribute/AU_val",
                           model.sal_loss2_attribute, count)
            # add_write_loss("Dis/All", model.loss_discriminator, count)
            # add_write_loss("Dis/Neu", model.loss_G_GANE+model.sal_loss2, count)
            # add_write_loss(
            #     "Dis/Attribute", model.loss_G_GANE_attribute+model.sal_loss2_attribute, count)
            # add_write_loss(
            #     "Gen/Neu", model.loss_GR, count)
            # add_write_loss(
            #     "Gen/Attribute", model.loss_generator_attribute, count)]

        torch.save(model.state_dict(), f"model_{epoch}.pth")


def main():
    for i, data in enumerate(dataset):
        #    pdb.set_trace()
        if i >= opt.how_many:
            break

        model.set_input(data)
        img_path = model.get_image_paths()
        model.test()

        img_path = model.get_image_paths()
    #    print('%04d: process image... %s' % (i, img_path))
    print("Done!")


if __name__ == '__main__':
    train()
