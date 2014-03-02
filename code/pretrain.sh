python main.py -n frelu --epochs 0 --training full
python main.py -n flog --epochs 0 --training full
python main.py -n fsp --epochs 0 --training full

for file in ../data/params/f*.npy; do
    cp $file ${file/_params/_pretrain_params};
done

