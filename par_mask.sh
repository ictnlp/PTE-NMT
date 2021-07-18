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