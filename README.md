# Are Transformers Effective for Time Series Forecasting? (AAAI 2023)
- Implementaion of the LTSF-DLinear Model Incorporating Future Past and Static Covariates
- Reference Paper [Are Transformers Effective for Time Series Forecasting?(AAAI 2023)](https://arxiv.org/pdf/2205.13504.pdf)

## LTSF-Linear Family
[image](./pics/Linear.png)
LTSF-Linear is a set of linear models.

Linear: It is just a one-layer linear model, but it outperforms Transformers.
* NLinear: To boost the performance of Linear when there is a distribution shift in the dataset, NLinear first subtracts the input by the last value of the sequence. Then, the input goes through a linear layer, and the subtracted part is added back before making the final prediction. The subtraction and addition in NLinear are a simple normalization for the input sequence.
* DLinear: It is a combination of a Decomposition scheme used in Autoformer and FEDformer with linear layers. It first decomposes a raw data input into a trend component by a moving average kernel and a remainder (seasonal) component. Then, two one-layer linear layers are applied to each component and we sum up the two features to get the final prediction. By explicitly handling trend, DLinear enhances the performance of a vanilla linear when there is a clear trend in the data.
Although LTSF-Linear is simple, it has some compelling characteristics:

* An O(1) maximum signal traversing path length: The shorter the path, the better the dependencies are captured, making LTSF-Linear capable of capturing both short-range and long-range temporal relations.
* High-efficiency: As each branch has only one linear layer, it costs much lower memory and fewer parameters and has a faster inference speed than existing Transformers.
* Interpretability: After training, we can visualize weights to have some insights on the predicted values.
Easy-to-use: LTSF-Linear can be obtained easily without tuning model hyper-parameters.
