
python3 cerberusdet/detect.py \
--img 640 --half \
--weights 'voc_obj365_animals_tableware_bs_best.pt' \
--source data/images \
--device 0 \
--hide-conf
