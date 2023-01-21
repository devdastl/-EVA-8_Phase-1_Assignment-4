# EVA-8_Phase-1_Assignment-4
This is the assignment of 4th session in phase-1 of EVA-8 from TSAI. 

## Introduction
### Objective:
Objective of this assignment is to build a CNN based network which will take [MNIST Handwritten Dataset](<http://yann.lecun.com/exdb/mnist/>) and will achieve targets as mentioned below.
- Upto `99.4%` test accuracy.
- Using total parameters count of under 10,000.
- Test accuracy should be consistant and need to be achieved till training reaches 15th epoch.

While achieving above mentioned target code needs to be writtern under certain conditions as mentioned below:
1. Target should be achieved in more then 3 steps in a gradual fashion. Means there should be more then 3 versions of the code where each code will be progressive in nature.
2. Each code version should have a target, result and analysis block showing what was the target of that code and what were the results and your analysis.

### Repository setup:
This repository contains 4 folders, each folder contains colab copy of notebook of the training code and a model architecture image. Below are the information about the following folders.
- step_1/ - This folder (as name suggest) contains colab copy of notebook of first step i.e. first code setup `step_1/EVA_assignment_4-step1.ipynb`. 
- step_2/ - This folder contains colab copy of notebook of second step i.e. second code setup `step_2/EVA_assignment_4-step2.ipynb`
- step_3/ - This folder contains colab copy of notebook of third step i.e. Third code setup `step_3/EVA_assignment_4-step3.ipynb`
- step_4/ - This folder contains colab copy of notebook of fourth step i.e. fourth code setup `step_4/EVA_assignment_4-step4.ipynb`
- Bonus_step/ - This assignment has a bonus part in which same target should be achieved in less then 8,000 parameters for bonus points. So this folder contains colab copy of notebook of that extra bonus step code setup `Bonus_step/EVA_assignment_4-Bonusstep.ipynb`

**NOTE: This means that required target is achieved in four steps and an extra step to achieve bonus target.**

## Dataset representation
As mentioned earlier, in this assignment we are using [MNIST Handwritten Digit Dataset](<http://yann.lecun.com/exdb/mnist/>) to train our CNN model to achieve mentioned target. 
Below is an image representing subset of the image data which we will be using in this assignment.

![Alt text](step_1/dataset_img.JPG?raw=true "model architecture")

## First code setup (step 1)
In this section we will look into the first code setup which is present in `step_1/EVA_assignment_4-step1.ipynb`. We will look into the target set for first code setup step, model architecture, result target, analysis and output logs.

### Target (step 1)
Below are the target for step1.
1. Get the setup correct and working. Because it is required to have basic working code for training and evaluation.
 - setup basic data transformations like normalization.
 - setup data loader for training and testing.
 - setup training and evaluation loop
2. Build CNN architecture/skeleton to have <10,000 parameters. Because assignment strongly mentions to have less then 10k parameters, so building larger CNN model is useless and will cause problem in follow-up code setup.

### Model architecture (step 1)
As mentioned in target that we will be targeting a model architecutre which will have less than 10k parameters. Below is an image of model architecture which achieve this target.
![Alt text](step_1/model_arch_step1.JPG?raw=true "model architecture")
Above image also contians inputs and outputs of each layer as well as **Receptive Field and it's calcuation**. Following are the explaination of terms used for **Receptive Field** calculation.
- jumpIN - pixel jump calculated for previous layer.
- jumpOUT - pixel jump calculated for current layer. Formula for jumpOUT is `jumpOUT = jumpIN x stride`
- stride - by how many pixel fliter is sliding our feature map.
- rIN - receptive field of previous layer.
- rOUT - receptive filed of current layer. Formula for rOUT is `rIN + (kernel_size - 1)xjumpIN`

### Result (step 1)
Below are the result achieved in the first code setup:
1. Total number of parameters - `9,734 (<10k)`
2. Training accuracy at 15th epoch - `98.81%`
3. Testing accuracy at 15th epoch - `98.62%`
4. Training accuracy at 20th epoch - `99.0%`
5. Testing accuracy at 20th epoch - `98.73%`

Also below is the graph generated after training:
![Alt text](step_1/result_graph_step1.JPG?raw=true "model architecture")

### Analysis (step 1)
Following are the analysis of this first code setup:

1. We build a CNN which is able to train under 10k parameter.
2. Highest train and test accuracy (20th epoch) is 99.0% and 98.73% resp. which is very less. Accuracy can be further improved.
3. Based on accuracy, model seems to be overfitting as training accuracy is larger then testing accruacy.

## Second code setup (step 2)
In this section we will look into the second code setup which is present in `step_2/EVA_assignment_4-step2.ipynb`. This is an interesting step where batch normalization is introduced and results looks good.

### Target (step 2)
Following are the targets for second code setup.
1. Improve overall train and test accuracy.
2. Improve model overfitting i.e. reduce difference between train and test accuracy.
3. Introduce very necessary component known as "batch normalization" in the CNN architecture

### Model architecture (step 2)
Below is an image of model architecture in second code setup.
![Alt text](step_2/model_arch_step2.JPG?raw=true "model architecture")

- Since only batch normalization is introduced, Receptive Field calculation and input-output shapes will be same hence not presented in above image.
- Also as represented, model architecture has four major convoluation blocks and one transition block. (This representation applies for all model mentioned here).

### Result (step 2)
Below are the results of second code setup.
1. Total number of parameters - `9,930 (<10k) (small increase due to Batch norm learnable mean and standerd deviation)
2. Training accuracy at 15th epoch - `99.55%`
3. Testing accuracy at 15th epoch - `99.36%`
4. Training accuracy at 20th epoch - `99.62%`
5. Testing accuracy at 20th epoch - `99.43%`

Below in an graph image produced from training-testing loss and accraucy:
![Alt text](step_2/result_graph_step2.JPG?raw=true "model architecture")

### Analysis (step 2)
Following are the analysis of this second code setup:
1. Over all accuracy of train and test dataset has been improved by alot (train: 98.81 to 99.55, test: 98.62 to 99.36)while using "Batch Normalization" in CNN architecture compared to the first setup.
2. "Batch Normalization" normalizes feature map across batches in each layer hence fixing the distribution of data.
3. There is still exist the problem of overfitting as training accuracy is still larger then testing accruacy although the gap is now smaller comare to first setup.
4. We didn't reach the target of 99.4% test accuracy in the second code setup.
5. Parameter counts increased by ~150 after adding batch norm because batch norm introduce new learnable parameters i.e. mean and std (alpha, beta).

## Third code setup (step 3)
In this section we will look into the third code setup which is present in `step_3/EVA_assignment_4-step3.ipynb`. This is also an interesting step because here we see effect of underfitting while training the same CNN.

### Target (step 3)
Following are the targets for third code setup.
1. Reduce the model overfitting.
2. Introduce "Drop Out" in CNN model. This will also helps with overfitting by randomly killing neurons in a layer while training the model and forcing other layers neuron not to focus on single neuron everytime.
3. Use image transformation to augment training images. Image Augmentation helps to reduce overfitting by forcing model to fit more images so that any kind of bias is ruled output.

### Model architecture (step 3)
Below is an image of model architecture in third code setup.
![Alt text](step_3/model_arch_step3.JPG?raw=true "model architecture")

- Since only Drop Out is introduced, Receptive Field calculation and input-output shapes will be same hence not presented in above image.

### Result (step 3)
Below are the results of third code setup.
1. Total number of parameters - `9,930 (<10k)`
1. Training accuracy at 15th epoch - `98.77%`
1. Testing accuracy at 15th epoch - `99.16%`
1. Training accuracy at 20th epoch - `98.84%`
1. Testing accuracy at 20th epoch - `99.26%`

Below in an graph image produced from training-testing loss and accraucy:
![Alt text](step_3/result_graph_step3.JPG?raw=true "model architecture")

### Analysis (step 3)
Following are the analysis of this third code setup:
1. Model is now underfitting i.e. testing accuracy is larger then training accuracy.
1. Due to larger underfitting accuracy dropped for both training and testing dataset. Not good for the model performance.
1. Due to "Drop Out" model could be suffering from excessive regularization which migh have impacted overall perfromance of the model as well as underfitting.
1. Accuracy is also fluctuating alot due to fluctuation in loss in later epochs. Maybe changing the learning rate in step can help to smooth out the decent.

## Fourth code setup (step 4)
In this section we will look into the fourth and **final** code setup which is present in `step_4/EVA_assignment_4-step4.ipynb`.

### Target (step 4)
Following are the targets for fourth code setup.
1. Improve underfitting of the model by setting Drop Out probablity to 0 i.e. we will not perform Drop Out while training in this code setup.
1. Improve accuracy/loss fluctuation by introducing step-wise learning rate decay using `StepLR()` under `torch.optim.lr_scheduler`. This is a pytorch module which takes optimizer, decay-rate and step-size to reduce the learning rate while training the model.

### Model architecture (step 4)
Below is an image of model architecture in fourth code setup.
![Alt text](step_4/model_arch_step4.JPG?raw=true "model architecture")

- In this architecture "Drop Out" has been removed hence there is no drop out in any layer in above image.
- Last 1x1 convolution does not has any activation, batch norm, etc. because last layer should not have anything.

### Result (step 4)
Below are the results of fourth code setup.
1. Total number of parameters - `9,930 (<10k)`
2. Training accuracy at 15th epoch - `99.48%`
3. Testing accuracy at 15th epoch - `99.49%`
4. Training accuracy at 20th epoch - `99.45%`
5. Testing accuracy at 20th epoch - `98.49%`

Below in an graph image produced from training-testing loss and accraucy:
![Alt text](step_4/result_graph_step4.JPG?raw=true "model architecture")

### Analysis (step 4)
Following are the analysis of this fourth code setup:
1. Setting Drop out to 0 (no Drop out) improves model accuracy for both train and test dataset. Means Drop Out was doing excessive regularization in third code setup.
1. Only data augmentation was enough to solve our overfitting problem in code setup 2.
1. By introducing Learning rate decay Model is giving consistant accuaracy for training and testing set.

#### **We reached our objective of consistant >99.4% accuracy under 10k parameters and under 15 epochs**
- Test accuracy at 15th epoch - 99.49%
- Consistant? - YES (consistantly hitting from 6th epoch onward till 20th)
- parameters - 9,930 (under 10k)

### Training log snippet (step 4)
```
EPOCH: 12
Loss=0.008368389680981636 Batch_id=937 Accuracy=99.47: 100%|██████████| 938/938 [00:49<00:00, 18.90it/s]

Test set: Average loss: 0.0173, Accuracy: 9947/10000 (99.47%)

EPOCH: 13
Loss=0.004195100627839565 Batch_id=937 Accuracy=99.50: 100%|██████████| 938/938 [00:50<00:00, 18.60it/s]

Test set: Average loss: 0.0170, Accuracy: 9948/10000 (99.48%)

EPOCH: 14
Loss=0.007462228648364544 Batch_id=937 Accuracy=99.47: 100%|██████████| 938/938 [00:49<00:00, 18.80it/s]

Test set: Average loss: 0.0169, Accuracy: 9952/10000 (99.52%)

EPOCH: 15
Loss=0.004266421776264906 Batch_id=937 Accuracy=99.48: 100%|██████████| 938/938 [00:49<00:00, 18.80it/s]

Test set: Average loss: 0.0169, Accuracy: 9949/10000 (99.49%)
```

## Bonus code setup (bonus step for bonus points)
In this section we will look into bonus code setup which will try to achieve target for bonus points.

### Target (bonus step)
This is an extra step where we will try to achieve consistant upto 99.4% accruacy under 8k parameters
1. Since our model is already achieving desired targets in 9,930 parameters. We can utilize same Architecture and reduce parameters count by playing around number of channels.
2. we want to reduce parameters from 9,930 to something less then 8,000 to get the additional point as mentioned in the assignment.
3. Lets also add another augmentation known as ColorJitter which will play around with the brightness, contrast, etc. This will allow us to learn more rich features from image.

### Model architecture (bonus step)
Below is an image of model architecture in bonus code setup.
![Alt text](Bonus_step/model_arch_bonus-step.JPG?raw=true "model architecture")

- Here we have changed number of channels in almost layers (refer to code `Bonus_step/EVA_assignment_4-BonusStep.ipynb` for more information)

### Result (bonus step)
Below are the results of bonus code setup.
1. Total number of parameters - 7,926 (<8k)
1. Training accuracy at 15th epoch - 99.26%
1. Testing accuracy at 15th epoch - 99.46%
1. Training accuracy at 20th epoch - 99.21%
1. Testing accuracy at 20th epoch - 98.46%

Below in an graph image produced from training-testing loss and accraucy:
![Alt text](Bonus_step/result_graph_bonus-step.JPG?raw=true "model architecture")

### Analysis (bonus step)
Following are the analysis of this bonus code setup:
1. By reducing the number of channels in the fourth code setup CNN, we were able to hit parameter count of 7,926 which is lesser then 8,000.
2. Reducing parameters count also reduces model complexity which lead to small underfitting as can be seen by test and train accuracy.

#### **We reached our objective of consistant >99.4% accuracy under 8k parameters and under 15 epochs**
- Test accuracy at 15th epoch - 99.49%
- Consistant? - YES (consistantly hitting from 10th epoch onward till 20th)
- parameters - 7,926 (under 8k)

### Training log snippet (bonus step)
```
EPOCH: 12
Loss=0.07796341925859451 Batch_id=937 Accuracy=99.26: 100%|██████████| 938/938 [01:17<00:00, 12.09it/s]

Test set: Average loss: 0.0197, Accuracy: 9946/10000 (99.46%)

EPOCH: 13
Loss=0.016041133552789688 Batch_id=937 Accuracy=99.23: 100%|██████████| 938/938 [01:17<00:00, 12.11it/s]

Test set: Average loss: 0.0197, Accuracy: 9949/10000 (99.49%)

EPOCH: 14
Loss=0.0026311047840863466 Batch_id=937 Accuracy=99.24: 100%|██████████| 938/938 [01:17<00:00, 12.06it/s]

Test set: Average loss: 0.0195, Accuracy: 9946/10000 (99.46%)

EPOCH: 15
Loss=0.007067004218697548 Batch_id=937 Accuracy=99.26: 100%|██████████| 938/938 [01:16<00:00, 12.31it/s]

Test set: Average loss: 0.0201, Accuracy: 9944/10000 (99.44%)
```
