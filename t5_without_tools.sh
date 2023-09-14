cur_time=$(date "+%Y-%m-%d_%H:%M:%S")
# echo $cur_time

mkdir outputs/t5_without_tools/$cur_time
mkdir codes
cp -r scripts t5_without_tools.sh codes
mv codes outputs/t5_without_tools/$cur_time

python scripts/t5_without_tools.py --gpu 4
cp -r logs results outputs/t5_without_tools/$cur_time
