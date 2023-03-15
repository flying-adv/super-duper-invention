# Text-Controlled GAN Inversion for Image Editing

### Install packages 
```
pip install --upgrade gdown
pip install wandb
pip install lpips
!pip install git+https://github.com/openai/CLIP.git
```

### Ninja is require to load C++
```
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
```

### Download pretrained models 

```
cd pretrained_models
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
gdown --fuzzy "https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing"
```

### Run the inference. Input image is stored in test folder. Additionally image directory can be changed from paths_config.py from configs.
```
python run_pti.py
```

results will be saved at output directory
