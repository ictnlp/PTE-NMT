# PTE-NMT
Source code for the NAACL 2021 long paper Pruning-then-Expanding Model for Domain Adaptation of Neural Machine Translation.

## Related code

Implemented based on [Fairseq-py](https://github.com/pytorch/fairseq), an open-source toolkit released by Facebook which was implemented strictly referring to [Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf).

## Requirements
This system has been tested in the following environment.
+ OS: Ubuntu 16.04.1 LTS 64 bits
+ Python version \>=3.7
+ Pytorch version \>=1.0

## Get started
- Build
```
python setup.py build develop
```

- Preprocess the training data. Pretrain the general-domain model with the general-domain data. Read [here](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-a-new-model) for more instructions.

- Evaluate the importance of the parameters and prune the general domain model.

```
bash par_mask.sh
```

or

```
#!/bin/bash 

# the pre-trained general-domain checkpoint
ckt=

# path to save the pruned checkpoint
save_ckt=

# path to save the mask matrix 
save_mask=

# prune ratio
ratio=0.3

python magnitude.py --pre-ckt-path $ckt --save-ckt-path $save_ckt \
            --save-mask-path $save_mask --prune-ratio $ratio
```

- Train the pruned model with knowledge distillation

```
bash train.kd.sh
```

or

```
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
```

- Generate the general-domain translation 

```
python generate.py {General-domain-data} --path $MODEL \
    --gen-subset test --beam 4 --batch-size 128 \
    --remove-bpe --lenpen {float} \
```

The length penalty is set as 1.4 for the zh-en experiments and 0.6 for the en-de and en-fr experiments.

- Fine-tuning the model 

```
bash train.ft.sh
```

or

```
#save dir
save=

# the last checkpoint after knowledge distillation
ckt=checkpoint_last.pt

# mask file
mask=

CUDA_VISIBLE_DEVICES=1  python3  train.py --ddp-backend=no_c10d  data-bin/{in-domain-data}\
    --arch transformer_wmt_en_de  --fp16 --reset-optimizer \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
       --mask-file $mask  --restore-file $ckt --lr-scheduler fixed \
          --lr 7.5e-5 --dropout 0.1\
           --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0\
        --max-tokens  4096  --save-dir checkpoints/$save   --save-interval 1 \
        --update-freq 1 --no-progress-bar --log-format json --log-interval 25 
```

- Generate the in-domain translation

```
python generate.py {In-domain-data} --path $MODEL \
    --gen-subset test --beam 4 --batch-size 128 \
    --remove-bpe --lenpen {float} \
```

## Citation
```
@inproceedings{GuFX21,
  author    = {Shuhao Gu and
               Yang Feng and
               Wanying Xie},
  title     = {Pruning-then-Expanding Model for Domain Adaptation of Neural Machine
               Translation},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of
               the Association for Computational Linguistics: Human Language Technologies,
               {NAACL-HLT} 2021, Online, June 6-11, 2021},
  pages     = {3942--3952},
  year      = {2021},
  url       = {https://www.aclweb.org/anthology/2021.naacl-main.308/},
}
```
























