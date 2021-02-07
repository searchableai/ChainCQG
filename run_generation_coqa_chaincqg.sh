CUDA_VISIBLE_DEVICES=0 python main_coqa_two_gpt.py --random_seed 1111 \
                                 --warmup_steps 100 \
                                 --learning_rate 1e-5 \
                                 --batch_size 1 \
                                 --gradient_accumulation_steps 1 \
                                 --num_train_epochs 10 \
                                 --do_valid \
                                 --model_size medium \
                                 --dataset_name coqa_two_gpt