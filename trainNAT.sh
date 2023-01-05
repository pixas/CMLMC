#!/usr/bin/env bash
python train.py "/nvme/jsy/data-bin/iwslt14_deen_jointdict" \
    --arch cmlm_transformer_iwslt_en_de \
    -s de \
    -t en \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --criterion nat_loss \
    --task translation_lev \
    --label-smoothing 0.1 \
    --noise random_mask \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr '1e-07' \
    --lr 0.0005 \
    --warmup-updates 10000 \
    --dropout 0.3 \
    --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --share-all-embeddings \
    --max-tokens 8192 \
    --max-epoch 300 \
    --fixed-validation-seed 7 \
    --fp16 \
    --save-dir /nvme/jsy/checkpoints/IWSLTdeen_raw_CMLM_benchmark/ \

python InferenceIWSLT_valid.py IWSLTdeen_raw_CMLM_benchmark 80 300

python train.py "/nvme/jsy/data-bin/iwslt14_deen_jointdict" \
   --arch cmlm_transformer_iwslt_en_de \
   -s de \
   -t en \
   --optimizer adam \
   --adam-betas '(0.9,0.98)' \
   --criterion nat_loss \
   --task translation_lev \
   --label-smoothing 0.1 \
   --noise random_mask \
   --lr-scheduler inverse_sqrt \
   --warmup-init-lr '1e-07' \
   --lr 0.0005 \
   --warmup-updates 30000 \
   --dropout 0.3 \
   --weight-decay 0.01 \
   --decoder-learned-pos \
   --encoder-learned-pos \
   --apply-bert-init \
   --share-all-embeddings \
   --max-tokens 1024 \
   --max-epoch 300 \
   --fixed-validation-seed 7 \
   --fp16 \
   --no-scale-embedding \
   --insertCausalSelfAttn \
   --concatPE \
   --selfcorrection 0 \
   --replacefactor 0.3 \
   --save-dir /nvme/jsy/checkpoints/IWSLTdeen_raw_CMLMC_L5D3_30k/ \

python InferenceIWSLT_valid.py IWSLTdeen_raw_CMLMC_L5D3_30k 150 300


srun -p NLP --gres=gpu:4 -N1 --quotatype=spot --async -o amlpbothself_amlpseqcross_w5.txt  python train.py "/mnt/petrelfs/jiangshuyang/data-bin/wmt14_deen_distill_jointdict" \
   --arch cmlm_transformer_wmt_en_de \
   -s de \
   -t en \
   --optimizer adam \
   --adam-betas '(0.9,0.98)' \
   --criterion nat_loss \
   --task translation_lev \
   --label-smoothing 0.1 \
   --noise random_mask \
   --lr-scheduler inverse_sqrt \
   --warmup-init-lr '1e-07' \
   --lr 0.0005 \
   --warmup-updates 40000 \
   --dropout 0.2 \
   --weight-decay 0.01 \
   --decoder-learned-pos \
   --encoder-learned-pos \
   --apply-bert-init \
   --share-all-embeddings \
   --max-tokens 16384 \
   --max-update 300000 \
   --fixed-validation-seed 7 \
   --fp16 \
   --keep-last-epochs 20 \
   --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe \
   --landmarks 16 \
   --amlp-activation 'softmax' \
   --encoder-self-attention-type 'covamlp2' \
   --decoder-cross-attention-type 'amlpseq' \
   --decoder-self-attention-type 'covamlp2' \
   --no-scale-embedding \
   --concatPE \
   --selfcorrection 0 \
   --replacefactor 0.3 \
   --save-dir /mnt/cache/jiangshuyang/checkpoints/WMTdeen_distill_CMLMC_L5D3_300k_covamlpbothself_amlpseqcross_w5/ --ddp-backend=no_c10d 



python InferenceWMT_valid.py WMTdeen_distill_CMLMC_L5D3_300k_covamlpbothself_amlpseqcross_w5 110 129  /mnt/petrelfs/jiangshuyang/checkpoints/

python generate.py /mnt/petrelfs/jiangshuyang/data-bin/wmt14_deen_distill_jointdict --gen-subset test --task translation_lev --path /mnt/petrelfs/jiangshuyang/checkpoints/WMTdeen_distill_CMLMC_L5D3_300k_ablate_nocausal_abc/checkpoint_best.pt --batch-size 128 --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --iter-decode-force-max-iter --iter-decode-with-beam 3 --remove-bpe --quiet

python generate.py /mnt/petrelfs/jiangshuyang/data-bin/wmt14_deen_distill_jointdict --gen-subset test --task translation_lev --path /mnt/petrelfs/jiangshuyang/checkpoints/WMTdeen_distill_CMLMC_L5D3_300k_ablate_nocausal/LEN3_bestmodel_bestvalid_ensemble10_epoch100_144.pt --batch-size 128 --iter-decode-max-iter 1 --iter-decode-eos-penalty 0 --iter-decode-force-max-iter --iter-decode-with-beam 3 --remove-bpe --quiet

srun -p NLP --gres=gpu:4 -N1 --quotatype=reserved python train.py "/mnt/petrelfs/jiangshuyang/data-bin/wmt14_ende_distill_jointdict" \
   --arch cmlm_transformer_wmt_en_de \
   -s en \
   -t de \
   --optimizer adam \
   --adam-betas '(0.9,0.98)' \
   --criterion nat_loss \
   --task translation_lev \
   --label-smoothing 0.1 \
   --noise random_mask \
   --lr-scheduler inverse_sqrt \
   --warmup-init-lr '1e-07' \
   --lr 0.0005 \
   --warmup-updates 40000 \
   --dropout 0.2 \
   --weight-decay 0.01 \
   --decoder-learned-pos \
   --encoder-learned-pos \
   --apply-bert-init \
   --share-all-embeddings \
   --max-tokens 16384 \
   --max-update 300000 \
   --max-epoch 250 \
   --fixed-validation-seed 7 \
   --fp16 \
   --no-scale-embedding \
   --concatPE \
   --keep-last-epochs 20 \
   --selfcorrection 0 \
   --replacefactor 0.3 \
   --landmarks 16 \
   --amlp-activation 'softmax' \
   --encoder-self-attention-type 'mha' \
   --decoder-cross-attention-type 'amlpseq' \
   --decoder-self-attention-type 'covamlp2' \
   --save-dir /mnt/petrelfs/jiangshuyang/checkpoints/WMTende_distill_CMLMC_L5D3_300k_bsz64k_covamlp_decoderself_amlpseqcross  --ddp-backend=no_c10d

python InferenceWMT_valid.py WMTende_distill_CMLMC_L5D3_300k_bsz64k_covamlp_decoderself_amlpseqcross 89 108 /mnt/petrelfs/jiangshuyang/checkpoints/ /mnt/petrelfs/jiangshuyang/checkpoints/BLEU/ /mnt/petrelfs/jiangshuyang/data-bin/wmt14_ende_distill_jointdict/
