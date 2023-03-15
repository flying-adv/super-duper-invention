import os

import torch
from tqdm import tqdm

from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
from PIL import Image
import torch
import clip


device = "cuda"


class CLIPLoss(torch.nn.Module):

    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=1024 // 32)
        self.tokenizer = clip.tokenize

    def forward(self, image, text):
        image = torch.nn.functional.interpolate(image,(1024,1024))
        text = self.tokenizer(text).to('cuda')
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity

    def get_text_embeddings(self,text):
        text = self.tokenizer(text).to('cuda')
        features = self.model.encode_text(text)
        return features


class MultiIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb,prompt):
        super().__init__(data_loader, use_wandb)
        device = 'cuda'

        self.clip_loss = CLIPLoss().to(device)
        self.prompt = prompt

    def train(self):
     
        text_samples = self.prompt
        self.G.synthesis.train()
        self.G.mapping.train()

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True
        w_pivots = []
        images = []

        for fname, image in self.data_loader:
            print(image.shape)
            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            image_name = fname[0]
            if hyperparameters.first_inv_type == 'w+':
                embedding_dir = f'{w_path_dir}/{paths_config.e4e_results_keyword}/{image_name}'
            else:
                embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            text_features = self.clip_loss.get_text_embeddings([text_samples]*image.shape[0])
            pred_w = text_features
            # w_pivot = self.get_inversion(w_path_dir, image_name, image)
            # if hyperparameters.use_last_w_pivots:
            #     w_pivot = self.load_inversions(w_path_dir, image_name)

            # if not hyperparameters.use_last_w_pivots or w_pivot is None: 
            w_pivot = self.calc_inversions(image, image_name,pred_w,text_samples,self.clip_loss)            
            w_pivots.append(w_pivot)
            images.append((image_name, image))
            self.image_counter += 1

        for i in tqdm(range(hyperparameters.max_pti_steps)):
            self.image_counter = 0

            for data, w_pivot in zip(images, w_pivots):
                image_name, image = data

                if self.image_counter >= hyperparameters.max_images_to_invert:
                    break

                real_images_batch = image.to(global_config.device)

                generated_images = self.forward(w_pivot)
                loss, l2_loss_val, loss_lpips,loss_id = self.calc_loss(generated_images, real_images_batch, image_name,
                                                            self.G, use_ball_holder, w_pivot)

                loss_text = self.clip_loss(generated_images,text_samples)
                # print(loss_text)
                loss = 1.2 * loss + 0.5 * loss_text
                # loss = 0.5 * loss + 1.2 * loss_text

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                global_config.training_step += 1
                self.image_counter += 1

        # if self.use_wandb:
        log_images_from_w(w_pivots, self.G, [image[0] for image in images],text_samples)
        self.restart_training()
        w_pivots = []
        images = []          
        # del w_pivots
        # del w_pivots
        # torch.save(self.G,
        #            f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_multi_id.pt')
