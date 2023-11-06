# Missing Well Log Reconstruction Using MWLT
This is a repository for the paper "Missing Well Log Reconstruction: A Sequence Self-attention Deep Learning Framework Based on Transformer".

## abstract
Well logging is a critical tool for reservoir evaluation and fluid identification. However, due to the expanding borehole, instrument failure, economic constraints, etc., some types of well logs are sometimes missing. Existing logging curve reconstruction methods based on empirical formulas and fully connected deep neural networks (FCDNN) can only consider point-to-point mapping relationships. Recurrently structured neural networks can consider multi-point correlation, but it is difficult to compute in parallel. To take into account the correlation between log sequences and achieve computational parallelism, we propose a novel deep learning framework for missing well log reconstruction based on a state-of-the-art transformer architecture (MWLT). The MWLT utilizes a self-attentive mechanism instead of a circular recursive structure to model the global dependencies of the inputs and outputs. To employ different usage requirements, we design the MWLT in three scales: small, base, and large, by adjusting the parameters in the network. 8609 samples from 209 wells in the Sichuan Basin, China, are utilized for training and validation, and two additional blind wells are used for testing. The data augmentation strategy with random starting points is implemented to increase the robustness of the model. The results show that our proposed MWLT achieves a significant improvement in accuracy over the conventional Gardner equation and FCDNN, both on the validation dataset and blind test wells. The MWLT-Large and MWLT-Base have lower prediction errors than MWLT-Small but require more training time. Two wells in the Songliao Basin, China, are employed to evaluate the cross-region generalization performance of proposed method. The generalizability test results demonstrate density logs reconstructed by MWLT remain the best match to the truth compared to other methods. The parallelizable MWLT automatically learns the global dependence of the subsurface reservoir, enabling efficient and advanced missing well log reconstruction performance.
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
Lin L, Wei H, Wu T, et al. Missing well-log reconstruction using a sequence self-attention deep-learning framework[J]. Geophysics, 2023, 88(6): D391-D410.
```

BibTex

```@article{lin2023missing,
  title={Missing well-log reconstruction using a sequence self-attention deep-learning framework},
  author={Lin, Lei and Wei, Hao and Wu, Tiantian and Zhang, Pengyun and Zhong, Zhi and Li, Chenglong},
  journal={Geophysics},
  volume={88},
  number={6},
  pages={D391--D410},
  year={2023},
  publisher={Society of Exploration Geophysicists}
}

```
