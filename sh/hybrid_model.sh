python pennylane-torch-implementation.py \
--model "h_sample" \
--d_lr 3e-2 \
--g_lr 2e-2 \
--Finite_diff_step 0.1 \
--number_of_averaged_samples 200 \
--batch_size 100 \
--dataset_size 10_000 \
\
--epoch 50 \
--dimensions 2 \
--d_layers 1 \
--g_layers 1 \