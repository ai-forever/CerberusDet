
python3 cerberusdet/train.py \
--img 640 --batch 32 \
--data data/voc_obj365.yaml \
--weights pretrained/yolov8x_state_dict.pt \
--cfg cerberusdet/models/yolov8x_voc_obj365.yaml \
--hyp data/hyps/hyp.cerber-voc_obj365.yaml \
--name voc_obj365_v8x \
--device 0 \
--mlflow-url localhost
