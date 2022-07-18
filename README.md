# FedVLN

In our paper "[FedVLN: Privacy-preserving Federated Vision-and-Language Navigation](https://arxiv.org/abs/2203.14936)", we study the privacy problem of embodied agent and proposed a Federated Vision-and-Language Navigation algorithm for seen environment training and unseen environment pre-exploration. We achieved comparable results on seen environmen training with centralized training and improved over both centralized pre-exploration and environment-based pre-exploration. 

We release the reproducible code here.

## Environment Installation

Python requirements: Need python3.6
```
pip install -r python_requirements.txt
```

Install Matterport3D simulators:
```
git submodule update --init --recursive 
sudo apt-get install libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev
mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
# Replace the above line with following if it doesn't work:
#   cmake -DOSMESA_RENDERING=ON ..
make -j8
```

Note: 
if some error messages like `double err = cv::norm(reference_image, state->rgb, CV_L2);` pop up, please just ignore them.
They are about test but would not affect the training agent.

## Pre-Computed Features
### ImageNet ResNet152

Download image features for environments:
```
mkdir img_features
wget https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip -P img_features/
cd img_features
unzip ResNet-152-imagenet.zip
```

### CLIP Features
Please download the CLIP-ViT features with this link:
```shell
wget https://nlp.cs.unc.edu/data/vln_clip/features/CLIP-ViT-B-32-views.tsv -P img_features
```

## Training RxR

### Data
Please download the pre-processed data with link:
```shell
wget https://nlp.cs.unc.edu/data/vln_clip/RxR.zip -P tasks
unzip tasks/RxR.zip -d tasks/
```

### Training the Fed CLIP-ViL agent
We provide scripts to train agents for them separately with our extracted CLIP features.

    ```shell
    name=agent_rxr_en_clip_vit_fedavg_new_glr12
    flag="--attn soft --train listener
      --featdropout 0.3
      --angleFeatSize 128
      --language en
      --maxInput 160
      --features img_features/CLIP-ViT-B-32-views.tsv
      --feature_size 512
      --feedback sample
      --mlWeight 0.4
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 200000 --maxAction 35
      --if_fed True
      --fed_alg fedavg
      --global_lr 12
      --comm_round 910
      --n_parties 60"

    mkdir -p snap/$name
    CUDA_VISIBLE_DEVICES=0 python3 rxr_src/train.py $flag --name $name
    ```
    Or you could simply run the script with the same content as above(we will use this in the following):
    ```shell
    bash run/agent_rxr_clip_vit_en_fedavg.bash
    ```
    
### Training Fed Envdrop agent
    ```shell
    bash agent_rxr_resnet152_fedavg.bash
    ```

## Training R2R

### Download the Data
Download Room-to-Room navigation data:
```
bash ./tasks/R2R/data/download.sh
```

### Train the Fed CLIP-ViL Agent
Run the script:
```
bash run/agent_clip_vit_fedavg.bash
```
It will train the agent and save the snapshot under snap/agent/. Unseen success rate would be around 46%.

### Augmented training
- Train the speaker
  ```
  bash run/speaker_clip_vit_fedavg.bash
  ```
  It will train the speaker and save the snapshot under snap/speaker/

- Augmented training:

  After pre-training the speaker and the agnet,
  ```
  bash run/bt_envdrop_clip_vit_fedavg.bash
  ```
  It will load the pre-trained agent and train on augmented data with environmental dropout.
  
### Training the Fed Envdrop agent
- Agent
  ```shell
  bash run/agent_fedavg.bash
  ```
- Fed Speaker + Aug training
  ```shell
  bash run/speaker_fedavg.bash
  bash run/bt_envdrop_fedavg.bash
  ```

### Fed pre-exploration


## Related Links
- CLIP-ViL: [paper](https://arxiv.org/abs/2107.06383), [code](https://github.com/clip-vil/CLIP-ViL/tree/master/CLIP-ViL-VLN)
- EnvDrop: [paper](https://arxiv.org/abs/1904.04195), [code](https://github.com/airsplay/R2R-EnvDrop)
- R2R Dataset: [paper](https://arxiv.org/pdf/1711.07280.pdf), [code](https://github.com/peteanderson80/Matterport3DSimulator)
- RxR Dataset: [paper](https://arxiv.org/abs/2010.07954), [code](https://github.com/google-research-datasets/RxR

## Reference
If you use FedVLN in your research or wish to refer to the baseline results published here, 
please use the following BibTeX entry. 

```shell
@article{zhou2022fedvln,
  title={FedVLN: Privacy-preserving Federated Vision-and-Language Navigation},
  author = {Zhou, Kaiwen and Wang, Xin Eric},
  journal={arXiv preprint arXiv:2203.14936},
  year={2022}
}
```
