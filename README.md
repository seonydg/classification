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
