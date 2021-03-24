import torch
from collections import OrderedDict
#from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
#import itertools
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torchvision
import os
from torch import autograd
import copy
from natsort import natsorted, ns


class Estimate(nn.Module):
    def __init__(self):
        super(Estimate, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, stride=1, padding=1)
        )

    def forward(self, x):

        x = F.tanh(self.features(x))
        return x


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def _compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
            torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.B = opt.batchSize

        self.netG = torch.nn.DataParallel(networks.define_G(opt.input_nc+20, opt.output_nc, opt.ngf,
                                                            opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids), device_ids=self.gpu_ids)

        self.netGN = torch.nn.DataParallel(networks.define_G(opt.input_nc+20, opt.output_nc, opt.ngf,
                                                             opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids), device_ids=self.gpu_ids)

     #   self.I_E=Estimate().cuda()

        if self.isTrain:
            use_sigmoid = opt.no_lsgan

            self.netDA = torch.nn.DataParallel(networks.define_D(opt.input_nc, opt.ndf,
                                                                 opt.which_model_netD,
                                                                 opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids), device_ids=self.gpu_ids)

        self.schedulers = []
        self.optimizers = []

        # torch.optim.Adam(self.netG.parameters(), #                                                lr=opt.lr, betas=(0, 0.999),amsgrad=False,weight_decay=0)
        self.optimizer_G = torch.optim.RMSprop(
            self.netG.parameters(), lr=0.0002, alpha=0.99, eps=1e-8)
        # torch.optim.Adam(self.netGN.parameters(), #torch.optim.RMSprop(self.netGN.parameters(), lr=opt.lr)#
        self.optimizer_GN = torch.optim.RMSprop(
            self.netGN.parameters(), lr=0.0002, alpha=0.99, eps=1e-8)

        # torch.optim.Adam(self.netDA.parameters(), #torch.optim.RMSprop(self.netDA.parameters(), lr=opt.lr)#
        self.optimizer_DA = torch.optim.RMSprop(
            self.netDA.parameters(), lr=0.0002, alpha=0.99, eps=1e-8)

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_GN)

     #   self.optimizers.append(self.optimizer_DA)
        # self.optimizers.append(self.optimizer_I)

    #    for optimizer in self.optimizers:
     #           self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_network(
                self.netG, 'G', opt.which_epoch, self.optimizer_G)
            self.load_network(self.netGN, 'GN',
                              opt.which_epoch, self.optimizer_GN)
        #    self.load_network(self.I_E, 'I_E', opt.which_epoch)

            if self.isTrain:
                self.load_network(self.netDA, 'DA', opt.which_epoch)

#        self.generator_test = copy.deepcopy(self.netG)
#        self.generator_testN = copy.deepcopy(self.netGN)
        if self.isTrain:
            #            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(
                use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1_F = torch.nn.L1Loss(size_average=False)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.criterionCE = torch.nn.CrossEntropyLoss()

            # initialize optimizers

           # self.optimizer_I = torch.optim.Adam(self.I_E.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999)) #0.1 in other program

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netDA)
        print('-----------------------------------------------')

    def set_input(self, input):

        input_A = input['source']
        input_B = input['target']

        P_B = input['AU_target']

        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda()
            input_B = input_B.cuda()
            P_B = P_B.cuda()

        self.input_A = input_A

        self.P_B = P_B

        self.image_paths = input['A_paths']  # if AtoB else 'B_paths']

    def train(self):
        # self.fake_B: generated neutral image
        # self.real_B: final Ground Truth
        # self.tar = pseudo GT for neutral image generated by G_A
        # self.netG = G_A
        # self.Neut = neutral Action unit [0.5,0.5,0.5, 0,0,....0]
        for i in range(0, self.P_B.size(1)):
            self.netG.zero_grad()
            self.netDA.zero_grad()
            self.netGN.zero_grad()
            # 念の為
            for optimizer in self.optimizers:
                optimizer.zero_grad()

            # 入力するAU
            self.param_B = self.P_B[0, i, :].unsqueeze(0)
            self.param_B = self.param_B.view(-1, 20).float()

            # 入力画像
            self.real_A = self.input_A[:, 0:3, :, :]
            self.real_A.requires_grad = True

            # 中立顔のAUを作成
            self.AUN = self.param_B.view(self.param_B.size(0), self.param_B.size(1), 1, 1).expand(
                self.param_B.size(0), self.param_B.size(1), 128, 128)/100000
            self.AUN[:, 0:3] = 0.5

            # 中立顔の作成
            self.fake_B = self.netGN(
                torch.cat([self.real_A, self.AUN], dim=1))

            # 本当かどうか、予測したAU
            # This is the discriminator for G_N, that predicts real/fake and AUs
            pred_fake, E_con = self.netDA(self.fake_B)

            # 識別に関する損失
            self.loss_G_GANE = self.criterionGAN(pred_fake, True)  # GAN loss

            # 予測したAUに関する損失
            self.sal_loss2 = (self.criterionL1(E_con, self.Neut)) * \
                self.opt.lambda_A  # AU regression loss

            # Facial attribute reconstruction lossの計算のため
            # とりあえず一回、中立顔のAUをNeutraizerじゃ無い方にいれて中立顔の生成をしている
            with torch.no_grad():
                self.tar = self.netG(torch.cat([self.fake_B, self.AUN], dim=1))
                # Here self.netG is the G_A in the equaltion.I calculate the ground truth for the neutral image as self.tar. Note that the
                # gradient is not passing to the G_A. Otherwise G_A will know what G_N is doing and can be able to generates
                # everything by itself making neutral image blank.**

            # Neutraierとそうでないので生成した中立顔が同一かの損失
            # (Facial attribute reconstruction loss)
            self.recon = self.criterionL1(self.fake_B, self.tar.detach(
            ))*self.opt.lambda_B  # calulated |G_N(L_s)-G_A(L_s)|

            # grey scale image for LightCNN
            self.fake_B_gray = self.fake_B[:, 0, :, :] * 0.299 + self.fake_B[:, 1, :, :] * \
                0.587 + self.fake_B[:, 2, :, :] * \
                0.114

            # LightCNNに入れた後のidentityは同じであるという損失
            # 中立顔と、入力した顔で計算
            self.recon_Light = self.criterionLight(self.fake_B_gray.unsqueeze(
                1), self.real_A_gray.detach())*0.5   # LightCNN loss

            # 入力するAUをGANに入れる形式に変換
            AUR = self.param_B.view(self.param_B.size(0), self.param_B.size(1), 1, 1).expand(
                self.param_B.size(0), self.param_B.size(1), 128, 128)

            # Again passing neutral image to G_A with final AUs to generate the reenacted output. (which is the actual GT)**
            # 目的とするAUで目的とする画像を生成
            self.fake_B_re = self.netG(torch.cat([self.fake_B, AUR], dim=1))
            # 再生成損失
            self.R = self.criterionL1(self.fake_B_re, self.real_B.detach(
            ))*self.opt.lambda_B   # calculate the recon loss
            # これでneutraizerの損失らしい？？？
            self.loss_GR = self.loss_G_GANE + self.sal_loss2 + \
                self.recon + self.recon_Light + self.R

            # neutraizer backward
            self.loss_GR.backward()
            self.optimizer_GN.step()

            # grey scale image for LightCNN
            self.fake_B_re_gray = self.fake_B_re[:, 0, :, :] * 0.299 + self.fake_B_re[:, 1, :, :] * \
                0.587 + self.fake_B_re[:, 2, :, :] * 0.114
            # LightCNNに入れた後のidentityは同じであるという損失
            # 生成顔と、正解データで計算
            self.recon_Light_attribute = self.criterionLight(self.fake_B_gray_re.unsqueeze(
                1), self.real_B_gray.detach())*0.5
            pred_fake_attribute, E_con_attribute = self.netDA(self.fake_B_re)
            self.loss_G_GANE_attribute = self.criterionGAN(
                pred_fake_attribute, False)
            # 予測したAUに関する損失
            self.sal_loss2_attribute = (self.criterionL1(E_con_attribute, self.param_B)) * \
                self.opt.lambda_A

            # 識別機に関する損失
            self.loss_discriminator = self.sal_loss2+self.sal_loss2_attribute + \
                self.loss_G_GANE+self.loss_G_GANE_attribute
            self.loss_discriminator.backward()
            self.optimizer_DA.step()

            # neutraizerじゃない方の生成器
            self.loss_generator_attribute = self.loss_G_GANE + self.sal_loss2 + \
                self.recon + self.recon_Light + self.R+self.sal_loss2_attribute + \
                self.loss_G_GANE_attribute
            self.loss_generator_attribute.backward()
            self.optimizer_G.step()

    def test(self):

        self.netG.eval()
        self.netGN.eval()
      #  self.I_E.eval()
#        pdb.set_trace()
        desti = os.path.dirname(self.image_paths[0])+'/'+self.opt.results_dir
   #     pdb.set_trace()
        if not os.path.isdir(desti):
            os.makedirs(desti)

            for i in range(0, self.P_B.size(1)):

                self.param_B = self.P_B[0, i, :].unsqueeze(0)
                self.param_B = self.param_B.view(-1, 20).float()

                self.real_A = self.input_A[:, 0:3, :, :]
                self.real_A.requires_grad = True

#                    I_p=self.I_E(self.real_A)

#                    self.param_A=self.param_A.view(-1,20).float()
                self.AUN = self.param_B.view(self.param_B.size(0), self.param_B.size(1), 1, 1).expand(
                    self.param_B.size(0), self.param_B.size(1), 128, 128)/100000

                self.AUN[:, 0:3] = 0.5

                self.fake_B = self.netGN(
                    torch.cat([self.real_A, self.AUN], dim=1))
                #####################ORIGINAL######################################################

                AUR = self.param_B.view(self.param_B.size(0), self.param_B.size(1), 1, 1).expand(
                    self.param_B.size(0), self.param_B.size(1), 128, 128)
                ###################################################################################

             #   I_f=self.I_E(self.fake_B)

                self.fake_B_recon = self.netG(
                    torch.cat([self.fake_B.data, AUR], dim=1))

                torchvision.utils.save_image(
                    (self.fake_B_recon*0.5)+0.5, desti+'/'+str(i)+'_re'+'.png')

#
            # put your video_path
            os.system(
                "ffmpeg -r 25 -i ./new_crop/results_video/%01d_re.png -vcodec mpeg4 -y movie.mp4")
            os.system('rm -r ./new_crop/results_video/')

    def get_image_paths(self):
        return self.image_paths
#        return self.ref_path

    def _compute_loss_D(self, estim, is_real):
        return -torch.mean(estim) if is_real else torch.mean(estim)
