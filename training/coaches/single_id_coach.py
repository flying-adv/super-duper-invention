import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
import torch
from PIL import Image
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

class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb,prompt):
        super().__init__(data_loader, use_wandb)
        device = 'cuda'
        cfg_file = "https://huggingface.co/lambdalabs/clip2latent/resolve/main/ffhq-sg2-510.yaml"
        checkpoint =  "https://huggingface.co/lambdalabs/clip2latent/resolve/main/ffhq-sg2-510.ckpt"        
        self.clip_loss = CLIPLoss().to(device)
        self.prompt = prompt

    def train(self):
        # fdfdff
        text_samples = self.prompt
        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)
        

        use_ball_holder = True

        for fname, image in tqdm(self.data_loader):
            image_name = fname[0]

            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = None

            skips=1
            n_samples_per_txt=1
            cond_scale=1.0
            show_progress=True
            truncation=1.0
            text_features = self.clip_loss.get_text_embeddings([text_samples]*image.shape[0])

            pred_w = text_features
            # if hyperparameters.use_last_w_pivots:
            #     w_pivot = self.load_inversions(w_path_dir, image_name)

            if not hyperparameters.use_last_w_pivots or w_pivot is None:
                w_pivot = self.calc_inversions(image, image_name,pred_w,text_samples,self.clip_loss)

            # w_pivot = w_pivot.detach().clone().to(global_config.device)
            w_pivot = w_pivot.to(global_config.device)

            
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)
            skips=1
            n_samples_per_txt=1
            cond_scale=1.0
            show_progress=True
            truncation=1.0
           
            w_pivot = w_pivot #+ pred_w
            print(w_pivot.shape)
            torch.save(w_pivot, f'{embedding_dir}/0.pt')

            for i in tqdm(range(hyperparameters.max_pti_steps)):

                generated_images = self.forward(w_pivot)
                loss, l2_loss_val, loss_lpips,loss_id = self.calc_loss(generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot)
                # for_clip_img = (generated_images * 127.5 + 128).clamp(0, 255)
                loss_text = self.clip_loss(generated_images,text_samples)
                # print(loss_text)
                loss = 1.2 * loss + 0.5 * loss_text

                self.optimizer.zero_grad()

                # if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                #     break
                
                loss.backward(retain_graph=True)
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                # if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                #     log_images_from_w([w_pivot], self.G, [image_name])

                global_config.training_step += 1
                log_images_counter += 1
            log_images_from_w([w_pivot], self.G, [image_name],text_samples)
            self.image_counter += 1

            torch.save(self.G,
                       f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pt')
