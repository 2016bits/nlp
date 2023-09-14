cur_time=$(date "+%Y-%m-%d_%H:%M:%S")
# echo $cur_time

mkdir outputs/execute/$cur_time
mkdir codes
cp -r scripts execute.sh codes
mv codes outputs/execute/$cur_time

python scripts/execute_program.py --gpu 4
cp -r logs results outputs/execute/$cur_time
