安装
3dgs
conda create -n gaussian_plant python=3.8 -y
conda activate gaussian_plant 

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips imageio scikit-image pandas IPython matplotlib seaborn open3d

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

追踪分割安装
cd Tracking-Anything-with-DEVA
pip install -e .
bash scripts/download_models.sh     # Download the pretrained models

git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO


数据处理
bash data/data_llff.sh path_to_video
目标识别
cd yolov9
python detect.py --weights weight/yolov9-c.pt --conf 0.1 --source path_to_video_images --device 0 --save-txt --save-conf

分割追踪
bash script/prepare_pseudo_label.sh plant/Volcano2 1 /datashare/dir_zhaoliang0/gaussian-grouping-3dmask2-rasterdepth-loss-hardsoftloss_0_Sparseloss_plant_yolo3/yolov9/runs/detect/exp8

深度数据
cd dpt
python get_depth_map_for_llff_dtu.py --root_path /datashare/dir_zhaoliang0/gaussian-grouping-3dmask2-rasterdepth-loss-hardsoftloss_0_Sparseloss_plant_yolo3/data/plant --scenes Volcano2,abc

训练
bash script/train.sh plant/Volcano2 1   记得修改train.py 156行gt_obj[gt_obj != 2] = 0的目标语义
                                        gaussian_model.py 620行 mask = (prob_obj3d_3dmask[2, :, :] > 0.3)
