# save dir
save=

# the pruned checkpoint
ckt=

# the general domain checkpoint
teacher_ckt=

# the absolute path to the mask file
mask=

CUDA_VISIBLE_DEVICES=0  python3  train.py --ddp-backend=no_c10d  data-bin/{in-domain-data}\
    --arch transformer_wmt_en_de  --fp16 --reset-optimizer \
	    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
	     --mask-file $mask  --restore-teacher-file $teacher_ckt --knowledge-distillation \
         --lr-scheduler fixed --restore-file $ckt  \
	        --lr 7.5e-5 --dropout 0.1\
           --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0\
		    --max-tokens  4096  --save-dir checkpoints/$save  --save-interval 1 \
		    --update-freq 1 --no-progress-bar --log-format json --log-interval 25  
		   
          
