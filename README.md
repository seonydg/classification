# classifications

## 1. AlexNet
- Cat & Dog : 분류
  1. dataset : Cat/Dog images(500여장)
  2. model : build

  
## 2. VGGNet
- Cat & Dog : 분류
  1. dataset : Cat/Dog images(500여장)
  2. model
     - build : vgg11 - vgg19(선택)
     - pretrained : vgg11 - vgg19(선택) + nn.AdaptiveAvgPool2d + fc(output=2)


## 3. VGG_19_model
- normal & Covid & Viral Pneumonia : 분류
1. dataset : 캐글(https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)
2. model : Convolutional Layer(pretrained vgg19) + nn.AdaptiveAvgPool2d + fc(output=3)


## 4.ResNet
- Cat & Dog : 분류
1. dataset : Cat/Dog images(500여장)
2. model : ResNet Build : ver_18, ver_34 - 기본블럭 사용(3X3) / ver_50, ver_101, ver_152 - 병목블럭 사용(1X1)(3X3)(1X1)
