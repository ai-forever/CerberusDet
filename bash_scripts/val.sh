
CUDA_VISIBLE_DEVICES=0 \
python3 cerberusdet/val.py \
--img 640 --half --batch-size 32 \
--weights 'voc_obj365_animals_v8x_best.pt' \
--data data/voc_obj365_animals.yaml \
--name voc_obj365_animals_v8x --device 0 \
--mlflow-url localhost

CUDA_VISIBLE_DEVICES=0 \
python3 cerberusdet/val.py \
--img 640 --half --batch-size 32 \
--weights 'voc_obj365_full_bs_best.pt' \
--data data/voc_obj365_full.yaml \
--name voc_obj365_full_v8x --device 0 \
--mlflow-url localhost

CUDA_VISIBLE_DEVICES=0 \
python3 cerberusdet/val.py \
--img 640 --half --batch-size 32 \
--weights 'voc_obj365_animals_tableware_bs_best.pt' \
--data data/voc_obj365_animals_tableware.yaml \
--name voc_obj365_animals_tableware_v8x --device 0 \
--mlflow-url localhost
