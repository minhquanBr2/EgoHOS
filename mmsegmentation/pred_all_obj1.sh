# inference
python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/seg_twohands_ccda.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/images \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_twohands \

python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb_ccda/twohands_to_cb_ccda.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb_ccda/best_mIoU_iter_76000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/images \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_cb

python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_cb_to_obj1_ccda/twohands_cb_to_obj1_ccda.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_cb_to_obj1_ccda/best_mIoU_iter_34000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/images \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_obj1

# visualize
python visualize.py \
       --mode twohands \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/images \
       --twohands_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_twohands \
       --vis_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_twohands_vis

python visualize.py \
       --mode cb \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/images \
       --cb_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_cb \
       --vis_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_cb_vis

python visualize.py \
       --mode twohands_obj1 \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/images \
       --twohands_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_twohands \
       --obj1_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_obj1 \
       --vis_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_obj1_vis