# inference
CUDA_VISIBLE_DEVICES=7 python predict_image.py \
       --config_file ./work_dirs/seg_twohands_ccda/seg_twohands_ccda.py \
       --checkpoint_file ./work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth \
       --img_dir ../testimages2/images \
       --pred_seg_dir ../testimages2/pred_twohands \

CUDA_VISIBLE_DEVICES=7 python predict_image.py \
       --config_file ./work_dirs/twohands_to_cb_ccda/twohands_to_cb_ccda.py \
       --checkpoint_file ./work_dirs/twohands_to_cb_ccda/best_mIoU_iter_76000.pth \
       --img_dir ../testimages2/images \
       --pred_seg_dir ../testimages2/pred_cb

CUDA_VISIBLE_DEVICES=7 python predict_image.py \
       --config_file ./work_dirs/twohands_cb_to_obj2_ccda/twohands_cb_to_obj2_ccda.py \
       --checkpoint_file ./work_dirs/twohands_cb_to_obj2_ccda/best_mIoU_iter_32000.pth \
       --img_dir ../testimages2/images \
       --pred_seg_dir ../testimages2/pred_obj2

# visualize
CUDA_VISIBLE_DEVICES=7 python visualize.py \
       --mode twohands \
       --img_dir ../testimages2/images \
       --twohands_dir ../testimages2/pred_twohands \
       --vis_dir ../testimages2/pred_twohands_vis

CUDA_VISIBLE_DEVICES=7 python visualize.py \
       --mode cb \
       --img_dir ../testimages2/images \
       --cb_dir ../testimages2/pred_cb \
       --vis_dir ../testimages2/pred_cb_vis

CUDA_VISIBLE_DEVICES=7 python visualize.py \
       --mode twohands_obj2 \
       --img_dir ../testimages2/images \
       --twohands_dir ../testimages2/pred_twohands \
       --obj2_dir ../testimages2/pred_obj2 \
       --vis_dir ../testimages2/pred_obj2_vis

