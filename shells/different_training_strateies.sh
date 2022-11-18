#! bash
function display() {
  YEL='\e[1;33m'
  YELB='\e[5;1;33m'
  NC='\033[25;15;0m' # No Color
  printf "${YELB}============================================================${NC}\n"
  printf "${YEL}\t ${1} ${NC}\n"
  printf "${YELB}============================================================${NC}\n"

}

MODELTYPE=resnet50

display "Training START"

display "From Scratch"
python train.py -r "datasets/final/train" --model-type=$MODELTYPE
python train.py -r "datasets/final/train" --model-type=$MODELTYPE --use-lr-scheduler
python train.py -r "datasets/final/train" --model-type=$MODELTYPE --use-auto-augment
python train.py -r "datasets/final/train" --model-type=$MODELTYPE --use-lr-scheduler --use-auto-augment

display "Train Whole Model"
python train.py -r "datasets/final/train" --model-type=$MODELTYPE --user-pretrained-weight
python train.py -r "datasets/final/train" --model-type=$MODELTYPE --user-pretrained-weight --use-lr-scheduler
python train.py -r "datasets/final/train" --model-type=$MODELTYPE --user-pretrained-weight --use-auto-augment
python train.py -r "datasets/final/train" --model-type=$MODELTYPE --user-pretrained-weight --use-lr-scheduler --use-auto-augment

display "Finetune Last Layer"
python train.py -r "datasets/final/train" --model-type=$MODELTYPE --user-pretrained-weight --finetune-last-layer
python train.py -r "datasets/final/train" --model-type=$MODELTYPE --user-pretrained-weight --finetune-last-layer --use-lr-scheduler
python train.py -r "datasets/final/train" --model-type=$MODELTYPE --user-pretrained-weight --finetune-last-layer --use-auto-augment
python train.py -r "datasets/final/train" --model-type=$MODELTYPE --user-pretrained-weight --finetune-last-layer --use-lr-scheduler --use-auto-augment

display "Training End"