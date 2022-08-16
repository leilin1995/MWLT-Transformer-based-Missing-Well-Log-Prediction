# MWLT-Transformer-based-Missing-Well-Log-Prediction
This is a repository for the paper "MWLT: Transformer-based Missing Well Log Prediction" (submit to JPSE)).

## abstract
Well logging is a critical tool for reservoir evaluation and fluid identification. However, due to the expanding borehole, instrument failure, economic constraints, etc., some types of well logs are sometimes missing. The conventional methods and fully connected deep neural networks (FCDNN) can only consider the mapping relationship between single depth points. Recurrently structured neural networks can consider multi-point correlation, but it is difficult to compute in parallel. To take into account the correlation between log sequences and achieve computational parallelism, we propose a network for predicting missing well logging based on a state-of-the-art transformer architecture (MWLT). The MWLT utilizes a self-attentive mechanism instead of a circular recursive structure to model the global dependencies of the inputs and outputs. To employ different usage requirements, we design the MWLT in three scales: small, base, and large, by adjusting the parameters in the network. We produce 8609 training and validation samples using 209 wells in the Sichuan Basin, China, and two additional blind wells for testing. The results show that our proposed MWLT achieves a significant improvement in accuracy over the conventional Gardner’s equation and FCDNN, both on the validation dataset and blind test wells. The MWLT-Large and MWLT-Base have lower prediction errors than MWLT-Small but require more training time. Two wells in the Songliao Basin, China, are used to test our proposed method's cross-site generalization performance. The results demonstrate that the density logs predicted by the MWLT still match the truth best compared to Gardner’s equation and FCDNN. The parallelizable MWLT automatically learns the global dependence of subsurface reservoirs, enabling efficient and advanced missing well logging prediction performance.
## Example
This is an example of a density profile prediction for a well in the Lianggaoshan workings, Sichuan Basin.
![image](https://github.com/leilin1995/MWLT-Transformer-based-Missing-Well-Log-Prediction/blob/master/main_article/Test_blind/Lianggaoshan/A1.png)

## Project Organization
The repository contains is organized logically according to the following figure, the results contain hyperparameter settings, training loss, evaluation metrics, well-trained model, and prediction results.
Test_blind contains two workspace test well examples, and you can use jupyter notebook to follow predict.ipynb to reproduce our results.
![image](https://github.com/leilin1995/MWLT-Transformer-based-Missing-Well-Log-Prediction/blob/master/main_article.png)


## Code

All training and test code are in the directory **code**.The code to calculate the evaluation metrics is in the **result**.

## Dataset

Due to data confidentiality agreements, we cannot directly release the training data. If you want to obtain the training data, please contact the authors and provide your personal information.

## Dependencies

* python 3.6.13
* pytorch 1.10.2
* torchvision 0.11.3
* numpy 1.19.2
* h5py 2.10.0
* pandas 1.1.5
* matplotlib 3.3.4

## Usage instructions
Download this project and build the dependency.
You can learn how to use this repository by looking at the examples in Test_blind.

## Citation

If you find this work useful in your research, please consider citing:

```

```

BibTex

```html

```
