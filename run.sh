python run.py sge secondary_structure \
    --batch_size 256 \
    --gradient_accumulation_steps 64 \
    --num_train_epochs 200 \
    --patience 5 \
    --seed 42