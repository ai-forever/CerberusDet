# to view the results:
# mlflow ui --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES="0" \
python cerberusdet/train.py \
   --data /usr/src/app/data/voc_obj365.yaml \
   --weights /usr/src/app/pretrained/yolov8x_state_dict.pt \
   --hyp /usr/src/app/data/hyps/hyp.cerber-voc_obj365.yaml \
   --cfg /usr/src/app/cerberusdet/models/yolov8x_voc_obj365.yaml \
   --img 640 \
   --batch-size 32 \
   --epochs 50 \
   --evolve 50 --evolve_per_task --evolver optuna \
   --name evolve_voc_obj365_v8x \
   --mlflow-url localhost \
   --params_to_evolve 'lr0,lrf,momentum,weight_decay,warmup_epochs,warmup_momentum,warmup_bias_lr,box,cls,dfl'
