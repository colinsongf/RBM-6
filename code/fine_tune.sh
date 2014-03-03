python main.py -n frelu1 --epochs 300 --training full --save --dropout --lcost &
python main.py -n flog1 --epochs 300 --training full --save --dropout --lcost &
python main.py -n fsp1 --epochs 300 --training full --save --dropout --lcost 

python main.py -n frelu2 --epochs 300 --training full --save --dropout &
python main.py -n flog2 --epochs 300 --training full --save --dropout &
python main.py -n fsp2 --epochs 300 --training full --save --dropout
