timestamp=$(date +%s)
mkdir -p logs/$timestamp
dir=logs/$timestamp

CUDA_VISIBLE_DEVICES=0 python inference.py --begin_idx 4000 --end_idx 6000 > $dir/log_0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python inference.py --begin_idx 6000 --end_idx 8000 > $dir/log_1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python inference.py --begin_idx 8000 --end_idx 10000 > $dir/log_2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python inference.py --begin_idx 10000 > $dir/log_3.txt 2>&1 &

wait

echo "All inference processes have completed."
