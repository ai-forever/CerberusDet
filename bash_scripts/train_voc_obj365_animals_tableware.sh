
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
python -m torch.distributed.launch --nproc_per_node 8 cerberusdet/train.py \
--img 640 --batch "4,4,40" \
--data data/voc_obj365_animals_tableware.yaml \
--weights pretrained/yolov8x_state_dict.pt \
--cfg cerberusdet/models/yolov8x_voc_obj365_animals_tableware.yaml \
--hyp data/hyps/hyp.cerber-voc_obj365_subsets.yaml \
--name voc_obj365_subsets_v8x \
--patience 10 --workers 4 --sync-bn
