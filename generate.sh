cur_time=$(date "+%Y-%m-%d_%H:%M:%S")
# echo $cur_time

mkdir outputs/generate/$cur_time
mkdir codes
cp -r scripts generate.sh codes
mv codes outputs/generate/$cur_time

python scripts/generate_program.py --gpu 3 --num_programs_per_example 3
cp -r logs results outputs/generate/$cur_time
