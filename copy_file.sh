#!/bin/bash
for ((i=0;i<=19;i++))
do 
aws s3 cp s3://syjiang_bucket/checkpoints/WMTdeen_distill_CMLMC_L5D3_300k_abc_all/checkpoint$((132+i)).pt /mnt/petrelfs/jiangshuyang/checkpoints/WMTdeen_distill_CMLMC_L5D3_300k_abc_all/
done