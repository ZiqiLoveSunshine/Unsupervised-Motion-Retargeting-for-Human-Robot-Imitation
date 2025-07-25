import torch
import numpy as np
import os
from torch import optim
from models.simple_GAN import Discriminator
from models.enc_and_dec import Encoder, Decoder
from models.utils import GAN_loss, ImagePool
from models.base_model import BaseModel
from option_parser import try_mkdir

class DEEP_model(BaseModel):
    def __init__(self, args, dataset):
        super(DEEP_model, self).__init__(args)
        self.dataset = dataset
        self.n_topologies = self.dataset.character_num
        self.args = args
        self.enc = Encoder(args, self.dataset.joint_topology).to(self.device)
        self.dec = Decoder(args, self.enc).to(self.device)
        self.dis = Discriminator(args, self.dataset.joint_topology).to(self.device)

        self.D_para = list(self.dis.parameters())
        self.G_para = list(self.enc.parameters()) +list(self.dec.parameters())
        #train
        if self.args.is_train:
            self.optimizerD = optim.Adam(self.D_para, args.learning_rate, betas=(0.9, 0.999))
            self.optimizerG = optim.Adam(self.G_para, args.learning_rate, betas=(0.9, 0.999))
            self.optimizers = [self.optimizerD, self.optimizerG]
            self.criterion_rec = torch.nn.MSELoss()
            self.criterion_gan = GAN_loss(args.gan_mode).to(self.device)
            self.criterion_cycle = torch.nn.L1Loss()
            self.fake_pool = ImagePool(args.pool_size)
        else:
            from option_parser import try_mkdir
            self.results_path = os.path.join(args.save_dir, 'results')
            try_mkdir(self.results_path)

    def discriminator_requires_grad_(self, requires_grad):
        for para in self.dis.parameters():
            para.requires_grad = requires_grad

    def set_input(self, motions):
        self.motions_input = motions

        if not self.is_train:
            self.motions_gt = []
            self.fake_list = []
            self.item_len = self.motions_input[-1]
            for i in range(1,self.n_topologies):
                self.motions_gt.append(self.motions_input[0][i])
                self.fake_list.append(self.motions_input[1][i])
            self.motions_input = self.motions_input[0][0], [self.motions_input[1][0]]

    def forward(self):
        self.offset_repr = self.dataset.offsets
        self.offset_repr = self.offset_repr.to(self.device)
        self.offset_repr.to(self.device)
        self.motion, self.offset_idx = self.motions_input # batch_size * frame * joint * features_in
        self.motion = self.motion.to(self.device)
        
        self.offset_idx = self.offset_idx[0]
        self.true_offset = self.offset_repr[self.offset_idx]
        self.true_offset_enc = torch.concat(self.motion.shape[1]*[self.true_offset.unsqueeze(-1).unsqueeze(0)], dim=0) 
        self.true_offset_enc = torch.concat(self.motion.shape[0]*[self.true_offset_enc.unsqueeze(0)], dim = 0) 
        # latent space
        self.latent = self.enc(self.motion, self.true_offset_enc) # batch_size * frame/nnn * joint *features_in 
        
        """
        all this is to choose an offset index which is differenct from input offset index for retargeting
        """
        if self.is_train:
            offset_idx = int(self.offset_idx)
            if offset_idx == 0:
                self.fake_idx = torch.randint(1, self.n_topologies, [1])
            elif offset_idx == self.n_topologies-1:
                self.fake_idx = torch.randint(0, offset_idx, [1])
            else:
                if torch.rand([1]) < 0.5:
                    self.fake_idx = torch.randint(0, offset_idx, [1])
                else:
                    self.fake_idx = torch.randint(offset_idx+1, self.n_topologies, [1])
        self.fake_offset = self.offset_repr[self.fake_idx]
        
        # retargeting
        self.fake_offset_latent = torch.concat(self.latent.shape[1]*[self.fake_offset.unsqueeze(-1)], dim = 0)
        self.fake_offset_latent = torch.concat(self.latent.shape[0]*[self.fake_offset_latent.unsqueeze(0)], dim = 0)
        self.fake_res = self.dec(self.latent, self.fake_offset_latent)

        # reconstruction
        self.true_offset_latent = torch.concat(self.latent.shape[1]*[self.true_offset.unsqueeze(-1).unsqueeze(0)], dim=0)
        self.true_offset_latent = torch.concat(self.latent.shape[0]*[self.true_offset_latent.unsqueeze(0)], dim=0)
        self.res = self.dec(self.latent, self.true_offset_latent)
        self.res_pos = torch.concat([self.res, self.true_offset_enc], dim = -1)
        self.motion_pos = torch.concat([self.motion, self.true_offset_enc], dim = -1)
        
        # fake latent
        self.fake_offset_enc = torch.concat(self.fake_res.shape[1]*[self.fake_offset.unsqueeze(-1)], dim=0)
        self.fake_offset_enc = torch.concat(self.fake_res.shape[0]*[self.fake_offset_enc.unsqueeze(0)], dim=0)
        self.fake_pos = torch.concat([self.fake_res, self.fake_offset_enc], dim = -1)
        self.fake_latent = self.enc(self.fake_res, self.fake_offset_enc)
       

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real.detach())
        loss_D_real = self.criterion_gan(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_gan(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) *0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        self.loss_D = 0
        fake = self.fake_pool.query(self.fake_pos)
        self.loss_D += self.backward_D_basic(self.dis, self.motion_pos, fake)
        self.loss_recoder.add_scalar('D_loss_gan', self.loss_D)

    def backward_G(self):
        # rec_loss
        self.rec_loss = self.criterion_rec(self.motion, self.res)
        self.loss_recoder.add_scalar('rec_loss', self.rec_loss)

        self.cycle_loss = self.criterion_cycle(self.latent, self.fake_latent)
        self.loss_recoder.add_scalar('cycle_loss', self.cycle_loss)
        # gan loss
        self.gan_loss = self.criterion_gan(self.dis(self.fake_pos), True)
        self.loss_recoder.add_scalar('gan_loss', self.gan_loss)
        # total loss
        self.loss_G_total = self.rec_loss * self.args.lambda_rec  + \
                            self.cycle_loss * self.args.lambda_cycle + \
                            self.gan_loss * self.args.lambda_gan
        self.loss_recoder.add_scalar('G_loss_total', self.loss_G_total)
        self.loss_G_total.backward()

    def optimize_parameters(self):
        self.forward()

        # update G
        print("update")
        self.discriminator_requires_grad_(False)
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()

        self.flag = False
        # update Ds
        if np.random.randn() >0.8:
            self.flag = True
            self.discriminator_requires_grad_(True)
            self.optimizerD.zero_grad()
            self.backward_D()
            self.optimizerD.step()

    def verbose(self):
        if self.flag:
            res = {'rec_loss': self.rec_loss.item(),
                    'G_loss_gan': self.gan_loss.item(),
                'cycle_loss': self.cycle_loss.item(),
                'D_loss_gan': self.loss_D.item(),
                'G_loss_total': self.loss_G_total.item()}
        else:
            res = {'rec_loss': self.rec_loss.item(),
                    'G_loss_gan': self.gan_loss.item(),
                'cycle_loss': self.cycle_loss.item(),
                'G_loss_total': self.loss_G_total.item()}
        return sorted(res.items(), key=lambda x: x[0])

    def save(self):
        from option_parser import try_mkdir
        path = os.path.join(self.model_save_dir, "epoch_"+str(self.epoch_cnt))
        try_mkdir(path)
        torch.save(self.enc.state_dict(), os.path.join(path, 'encoder.pt'))
        torch.save(self.dec.state_dict(), os.path.join(path, 'decoder.pt'))
        torch.save(self.dis.state_dict(), os.path.join(path, 'discriminator.pt'))
        print('Save at {} succeed!'.format(path))

        for i, optimizer in enumerate(self.optimizers):
            try_mkdir(os.path.join(self.model_save_dir,"optimizers"))
            file_name = os.path.join(self.model_save_dir, 'optimizers/{}/{}.pt'.format(self.epoch_cnt, i))
            try_mkdir(os.path.split(file_name)[0])
            torch.save(optimizer.state_dict(), file_name)
        loss_path = os.path.join(self.model_save_dir,"loss")
        try_mkdir(loss_path)
        loss_path = os.path.join(self.model_save_dir,"loss/")
        self.loss_recoder.save(loss_path)

    def load(self, epoch=None):
        path = os.path.join(self.model_save_dir, 'epoch_{}'.format(epoch))
        print('loading from', path)
        if not os.path.exists(path):
            raise Exception('Unknown loading path')
        print('loading from epoch {}......'.format(epoch))
        self.enc.load_state_dict(torch.load(os.path.join(path, 'encoder.pt'),
                                                     map_location=self.args.cuda_device))
        self.dec.load_state_dict(torch.load(os.path.join(path, 'decoder.pt'),
                                                map_location=self.args.cuda_device))
        # self.dis.load_state_dict(torch.load(os.path.join(path, 'discriminator.pt'),
        #                                         map_location=self.args.cuda_device))
        if self.is_train:
            for i, optimizer in enumerate(self.optimizers):
                file_name = os.path.join(self.model_save_dir, 'optimizers/{}/{}.pt'.format(epoch, i))
                optimizer.load_state_dict(torch.load(file_name))
        self.epoch_cnt = epoch

    def compute_test_result(self):
        all_err = []
        # self.all_recons = []
        for i in self.fake_list:
            self.fake_idx = [i]
            self.forward()
            _,__,joint_num, fea = self.fake_res.shape
            gt = self.motions_gt[i-1].reshape((-1, joint_num, fea))[:self.item_len,...]
            fake = self.fake_res.reshape((-1, joint_num, fea))[:self.item_len,...]
            err = torch.sqrt((gt-fake)**2)
            err = torch.mean(err)
            all_err.append(err)
        all_err = torch.tensor(all_err)
        print("all_err: ", all_err.mean(), all_err)
        return all_err.mean()

    def get_result(self):
        with torch.no_grad():
            self.all_recons = []
            for i in self.fake_list:
                self.fake_idx = [i]
                self.forward()
                _,__,joint_num, fea = self.fake_res.shape
                print("data len: ", self.item_len)
                gt = self.motions_gt[i-1].reshape((-1, joint_num, fea))[:self.item_len,...]
                fake = self.fake_res.reshape((-1, joint_num, fea))[:self.item_len,...]
                res_gt = self.motion.reshape((-1, joint_num, fea))[:self.item_len,...]
                res = self.res.reshape((-1, joint_num, fea))[:self.item_len,...]
                self.all_recons.append([gt, fake, res_gt, res])