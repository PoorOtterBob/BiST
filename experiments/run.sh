### LargeST ###
# python experiments/bist/main.py --dataset SD --kernel_size 3 --device cuda:0 --bs 64 --model_name bist --core 8 --seq_len 12 --horizon 12 --years 2017_2018_2019_2020_2021

# python experiments/bist/main.py --dataset GBA --kernel_size 3 --device cuda:0 --bs 64 --model_name bist --core 24 --seq_len 12 --horizon 12 --years 2017_2018_2019_2020_2021

# python experiments/bist/main.py --dataset GLA --kernel_size 3 --device cuda:0 --bs 64 --model_name bist --core 32 --seq_len 12 --horizon 12 --years 2017_2018_2019_2020_2021

# python experiments/bist/main.py --dataset CA --kernel_size 3 --device cuda:0 --bs 64 --model_name bist --core 64 --seq_len 12 --horizon 12 --years 2017_2018_2019_2020_2021



### XTraffic ###
# python experiments/bist/main.py --dataset XTraffic --kernel_size 3 --device cuda:0 --bs 64 --model_name bist --core 128 --seq_len 12 --horizon 12



### XXLTraffic ###
# python experiments/bist/main.py --dataset XXLTraffic --kernel_size 25 --device cuda:0 --bs 64 --model_name bist --core 3 --seq_len 96 --horizon 96 --years hour

# python experiments/bist/main.py --dataset XXLTraffic --kernel_size 25 --device cuda:0 --bs 64 --model_name bist --core 3 --seq_len 96 --horizon 192 --years hour

# python experiments/bist/main.py --dataset XXLTraffic --kernel_size 25 --device cuda:0 --bs 64 --model_name bist --core 3 --seq_len 96 --horizon 336 --years hour

# python experiments/bist/main.py --dataset XXLTraffic --kernel_size 25 --device cuda:0 --bs 64 --model_name bist --core 3 --seq_len 96 --horizon 96 --years day

# python experiments/bist/main.py --dataset XXLTraffic --kernel_size 25 --device cuda:0 --bs 64 --model_name bist --core 3 --seq_len 96 --horizon 192 --years day

# python experiments/bist/main.py --dataset XXLTraffic --kernel_size 25 --device cuda:0 --bs 64 --model_name bist --core 3 --seq_len 96 --horizon 336 --years day



### Additional datasets ###
# python experiments/bist/main.py --dataset PeMS03 --kernel_size 3 --device cuda:0 --bs 64 --model_name bist --core 48 --seq_len 12 --horizon 12

# python experiments/bist/main.py --dataset PeMS04 --kernel_size 3 --device cuda:0 --bs 64 --model_name bist --core 32 --seq_len 12 --horizon 12

# python experiments/bist/main.py --dataset PeMS07 --kernel_size 3 --device cuda:0 --bs 64 --model_name bist --core 96 --seq_len 12 --horizon 12

# python experiments/bist/main.py --dataset PeMS08 --kernel_size 3 --device cuda:0 --bs 64 --model_name bist --core 8 --seq_len 12 --horizon 12

# python experiments/bist/main.py --dataset METR-LA --kernel_size 3 --device cuda:0 --bs 64 --model_name bist --core 8 --seq_len 12 --horizon 12

# python experiments/bist/main.py --dataset KnowAir --kernel_size 5 --device cuda:0 --bs 64 --model_name bist --core 3 --seq_len 24 --horizon 24
