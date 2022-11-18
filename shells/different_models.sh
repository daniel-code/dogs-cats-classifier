#! bash
function display() {
  YEL='\e[1;33m'
  YELB='\e[5;1;33m'
  NC='\033[25;15;0m' # No Color
  printf "${YELB}============================================================${NC}\n"
  printf "${YEL}\t ${1} ${NC}\n"
  printf "${YELB}============================================================${NC}\n"

}

display "Training START"

for MODELTYPE in resnet18 resnet34 resnet50 resnet101 resnext50_32x4d resnext101_32x8d swin_t swin_s swin_b; do
  display $MODELTYPE
  python train.py -r "datasets/final/train" --model-type=$MODELTYPE --use-lr-scheduler --user-pretrained-weight --use-auto-augment --finetune-last-layer
done

display "Training End"
