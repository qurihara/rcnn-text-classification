rm data.txt
node process.js data/hashire_merosu.txt data.txt 0
node process.js data/rashomon.txt data.txt 1
python train.py train data.txt --use-gpu
