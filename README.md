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
- `step_1/` - This folder (as name suggest) contains colab copy of notebook of first step i.e. first code setup `step_1/EVA_assignment_4-step1.ipynb`. 
- `step_2/` - This folder contains colab copy of notebook of second step i.e. second code setup `step_2/EVA_assignment_4-step2.ipynb`
- `step_3/` - This folder contains colab copy of notebook of third step i.e. Third code setup `step_3/EVA_assignment_4-step3.ipynb`
- `step_4/` - This folder contains colab copy of notebook of fourth step i.e. fourth code setup `step_4/EVA_assignment_4-step4.ipynb`
- `Bonus_step/` - This assignment has a bonus part in which same target should be achieved in less then 8,000 parameters for bonus points. So this folder contains colab copy of notebook of that extra bonus step code setup `Bonus_step/EVA_assignment_4-Bonusstep.ipynb`

This means that required target is achieved in four steps and an extra step to achieve bonus target.

## Dataset representation
As mentioned earlier, in this assignment we are using [MNIST Handwritten Digit Dataset](<http://yann.lecun.com/exdb/mnist/>) to train our CNN model to achieve mentioned target. 
Below is an image representing subset of the image data which we will be using in this assignment.
![Alt text](step_1/dataset_img.JPG?raw=true "model architecture")

## First code setup (step 1)
In this section we will look into the first code setup which is present in `step_1/EVA_assignment_4-step1.ipynb`. We will look into the target set for first code setup step, model architecture, result target, analysis and output logs.
### Target
Below are the target for step1.
1. Get the setup correct and working. Because it is required to have basic working code for training and evaluation.
 - setup basic data transformations like normalization.
 - setup data loader for training and testing.
 - setup training and evaluation loop
2. Build CNN architecture/skeleton to have <10,000 parameters. Because assignment strongly mentions to have less then 10k parameters, so building larger CNN model is useless and will cause problem in follow-up code setup.

### Model architecture
As mentioned in target that we will be targeting a model architecutre which will have less than 10k parameters. Below is an image of model architecture which achieve this target.
![Alt text](step_1/model_arch_step1.JPG?raw=true "model architecture")
Above image also contians inputs and outputs of each layer as well as **Receptive Field and it's calcuation**. Following are the explaination of terms used for **Receptive Field** calculation.
- jumpIN - pixel jump calculated for previous layer.
- jumpOUT - pixel jump calculated for current layer. Formula for jumpOUT is `jumpOUT = jumpIN x stride`
- stride - by how many pixel fliter is sliding our feature map.
- rIN - receptive field of previous layer.
- rOUT - receptive filed of current layer. Formula for rOUT is `rIN + (kernel_size - 1)xjumpIN`

### Result
Below are the result achieved in the first code setup:
1. Total number of parameters - `9,734 (<10k)`
2. Training accuracy at 15th epoch - `98.81%`
3. Testing accuracy at 15th epoch - `98.62%`
4. Training accuracy at 20th epoch - `99.0%`
5. Testing accuracy at 20th epoch - `98.73%`

Also below is the graph generated after training:
![Alt text](step_1/result_graph_step1.JPG?raw=true "model architecture")

### Analysis
Following are the analysis of this first code setup:

1. We build a CNN which is able to train under 10k parameter.
2. Highest train and test accuracy (20th epoch) is 99.0% and 98.73% resp. which is very less. Accuracy can be further improved.
3. Based on accuracy, model seems to be overfitting as training accuracy is larger then testing accruacy.

## Second code setup (step 2)
In this section we will look into the second code setup which is present in `step_2/EVA_assignment_4-step2.ipynb`. This is an interesting step where batch normalization is introduced and results looks good.
### Target
Following are the targets for second code setup.
1. Improve overall train and test accuracy.
2. Improve model overfitting i.e. reduce difference between train and test accuracy.
3. Introduce very necessary component known as "batch normalization" in the CNN architecture

### Model architecture
Below is an image of model architecture in second code setup.
![Alt text](step_2/model_arch_step2.JPG?raw=true "model architecture")

- Since only batch normalization is introduced, Receptive Field calculation and input-output shapes will be same hence not presented in above image.
- Also as represented, model architecture has four major convoluation blocks and one transition block. (This representation applies for all model mentioned here).

### Result
Below are the results of second code setup.
1. Total number of parameters - `9,930 (<10k) (small increase due to Batch norm learnable mean and standerd deviation)
2. Training accuracy at 15th epoch - `99.55%`
3. Testing accuracy at 15th epoch - `99.36%`
4. Training accuracy at 20th epoch - `99.62%`
5. Testing accuracy at 20th epoch - `99.43%`

Below in an graph image produced from training-testing loss and accraucy:
![Alt text](step_2/result_graph_step2.JPG?raw=true "model architecture")

### Analysis
Following are the analysis of this second code setup:
1. Over all accuracy of train and test dataset has been improved by alot (train: 98.81 to 99.55, test: 98.62 to 99.36)while using "Batch Normalization" in CNN architecture compared to the first setup.
2. "Batch Normalization" normalizes feature map across batches in each layer hence fixing the distribution of data.
3. There is still exist the problem of overfitting as training accuracy is still larger then testing accruacy although the gap is now smaller comare to first setup.
4. We didn't reach the target of 99.4% test accuracy in the second code setup.
5. Parameter counts increased by ~150 after adding batch norm because batch norm introduce new learnable parameters i.e. mean and std (alpha, beta).

## Third code setup (step 3)


