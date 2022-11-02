# Performance <!-- omit in toc -->

- [Different Training Strategies](#different-training-strategies)
- [Different Models](#different-models)
- [Conclusion](#conclusion)

# Different Training Strategies

Compare different training strategies performance

```commandline
bash scripts/different_training_strateies.sh
```

- model-type: resnet50
- batch-size: 16
- max-epochs: 10
- seed: 168
- image-size: (256, 256)

|      Strategies      | Pretrained Weight | OneCycle | AutoAugment |  Accuracy  | Precision | Recall | AUCROC |
|:--------------------:|:-----------------:|:--------:|:-----------:|:----------:|:---------:|:------:|:------:|
|     From scratch     |                   |          |             |   0.8852   |   0.9579  | 0.8092 | 0.9718 |
|                      |                   |     V    |             |   0.9416   |   0.9223  | 0.9605 | 0.9877 |
|                      |                   |          |      V      |   0.8932   |   0.8307  | 0.9848 | 0.9845 |
|                      |                   |     V    |      V      |   0.9360   |   0.9256  | 0.9489 | 0.9844 |
|  Train Whole Model   |         V         |          |             |   0.9784   |   0.9789  | 0.9777 | 0.9990 |
|                      |         V         |     V    |             |   0.9892   |   0.9855  | 0.9927 | 0.9995 |
|                      |         V         |          |      V      |   0.9828   |   0.9781  | 0.9849 | 0.9996 |
|                      |         V         |     V    |      V      |   0.9920   |   0.9909  | 0.9931 | 0.9999 |
| Fine-tune Last Layer |         V         |          |             |   0.9928   |   0.9901  | 0.9959 | 0.9998 |
|                      |         V         |     V    |             |   0.9912   |   0.9864  | 0.9957 | 0.9997 |
|                      |         V         |          |      V      | **0.9948** |   0.9909  | 0.9978 | 0.9999 |
|                      |         V         |     V    |      V      |   0.9944   |   0.9901  | 0.9978 | 0.9999 |

# Different Models

Compare different models performance

```commandline
bash scripts/different_models.sh
```

- batch-size: 16
- max-epochs: 10
- seed: 168
- image-size: (256, 256)
- --use-lr-scheduler
- --user-pretrained-weight
- --use-auto-augment
- --finetune-last-layer

| Models           |  Accuracy  | Precision | Recall | AUCROC |
|------------------|:----------:|:---------:|:------:|:------:|
| resnet18         |   0.9844   |   0.9843  | 0.9846 | 0.9994 |
| resnet34         |   0.9832   |   0.9706  | 0.9972 | 0.9995 |
| resnet50         |   0.9944   |   0.9901  | 0.9978 | 0.9999 |
| resnet101        |   0.9964   |   0.9951  | 0.9979 | 1.0000 |
| resnext50_32x4d  |   0.9932   |   0.9917  | 0.9947 | 0.9998 |
| resnext101_32x8d |   0.9944   |   0.9902  | 0.9984 | 0.9999 |
| swin_t           |   0.9940   |   0.9923  | 0.9966 | 0.9999 |
| swin_s           |   0.9964   |   0.9952  | 0.9979 | 1.0000 |
| **swin_b**       | **0.9976** |   0.9961  | 0.9993 | 1.0000 |

# Conclusion

Training Strategies are evaluated in [different training strategies](#different-training-strategies).
With **IMAGENET1K_V1** pre-trained weight, the accuracy boosts by 9.3%. 
Only fine-tuning the last layer improves the accuracy by 10.7%.

The [OneCycle learning rate policy](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) increases model performance except in fine-tuning last layer.
The [AutoAugment](https://pytorch.org/vision/main/generated/torchvision.transforms.AutoAugment.html) boost accuracy 0.2% ~ 0.8%.

[Different models](#different-models) are evaluated, and all training strategies are applied.
In similar architecture, ResNet, ResNext, and Swin, the bigger model is, the higher accuracy is.
In dogs-cats datasets, the accuracy of ResNext is slightly lower than ResNet in the same depth version.
