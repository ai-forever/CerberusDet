
python3 cerberusdet/val.py \
--img 640 --half --batch-size 32 \
--weights 'voc_obj365_v8x_best.pt' \
--data data/voc_obj365.yaml \
--name voc_obj365_v8x --device 0 \
--mlflow-url localhost
