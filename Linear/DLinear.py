import torch
import torch.nn as nn

class _MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        if kernel_size % 2 == 0:
            self.padding_size_left = kernel_size // 2 - 1
            self.padding_size_right = kernel_size // 2

        else:
            self.padding_size_left = (kernel_size - 1) // 2
            self.padding_size_right = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        
    def forward(self, x):
        '''
        input 차원 (batch size, input length, n_variables)
        '''
        front = x[:, 0:1, :].repeat(1, self.padding_size_left, 1)
        end = x[:, -1:, :].repeat(1, self.padding_size_right, 1)
        x = torch.cat([front, x, end], dim=1)
        # (batch_size, n_variables, input_length) -> avg로 moving avg 구함
        x = self.avg(x.permute(0, 2, 1))
        # (batch_size, input_legnth, n_variables)
        x = x.permute(0, 2, 1)
        return x

class _SeriesDecomp(nn.Module):
    '''
    계절성 decomposition block
    '''
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size, stride=1)
    
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class DLinearModel(nn.Module):
    def __init__(
            self,
            config,
            shared_weights,
            const_init,
            **kwargs,
    ):
        """
        PyTorch module implementing the DLinear architecture.

        Parameters
        ----------
        input_dim
            The number of input components (target + optional covariate + historical future covariates) // 일별 판매량, 일별 매출 2
        output_dim
            Number of output components in the target // 일별 판매량 1
        future_cov_dim
            Number of components in the future covariates // 년도, 월, 일 3
        static_cov_dim
            Dimensionality of the static covariates // 대분류, 중분류, 소분류, 브랜드 4
        shared_weights
            Whether to use shared weights for the components of the series.
            ** Ignores covariates when True. **
        kernel_size
            The size of the kernel for the moving average // 논문의 default 25
        const_init
            Whether to initialize the weights to 1/in_len
        **kwargs
            all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        y of shape `(batch_size, output_chunk_length, target_size/output_dim)`
        """

        super().__init__(**kwargs)
        self.input_chunk_length = config["input_chunk_length"]
        self.output_chunk_length = config["output_chunk_length"]
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.future_cov_dim = config["future_cov_dim"]
        self.static_cov_dim = config["static_cov_dim"]
        self.const_init = const_init

        # Decomposition Kernel Size
        self.decomposition = _SeriesDecomp(config["kernel_size"])
        self.shared_weights = shared_weights

        def _create_linear_layer(in_dim, out_dim):
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True)  # ReLU 활성화 함수 추가
            )
            if self.const_init:
                layer[0].weight = nn.Parameter(
                    (1.0 / in_dim) * torch.ones(layer[0].weight.shape)
                )
            return layer

        
        if self.shared_weights:                                         # False로 할 거임
            layer_in_dim = self.input_chunk_length
            layer_out_dim = self.output_chunk_length
        else:
            layer_in_dim = self.input_chunk_length * self.input_dim     # 90일 * (일별 판매량, 일별 매출)
            layer_out_dim = self.output_chunk_length * self.output_dim  # 21일 * (일별 판매량)

            layer_in_dim_static_cov = self.output_dim * self.static_cov_dim # 1 * (대분류, 중분류, 소분류, 브랜드)
        
        self.linear_seasonal = _create_linear_layer(layer_in_dim, layer_out_dim)
        self.linear_trend = _create_linear_layer(layer_in_dim, layer_out_dim)

        if self.future_cov_dim != 0:
            # future covariates layer acts on time steps independently
            self.linear_fut_cov = _create_linear_layer(                     # 년도, 월, 일 // 안주는게 나을 수도
                self.future_cov_dim, self.output_dim
            )
        if self.static_cov_dim != 0:
            self.linear_static_cov = _create_linear_layer(
                layer_in_dim_static_cov, layer_out_dim
            )
    
    def forward(self, x_in):
        """
        x_in
            comes as tuple '(x_past, x_future, x_static)' 각 각의 input dimension은
            '(n_samples, n_time_steps, n_variables)"
        """
        x, x_future, x_static = x_in
        batch, _, _ = x.shape

        if self.shared_weights:
            # discard covariates, to ensure that in_dim == out_dim
            x = x[:, :, : self.output_dim]

            # extract trend
            res, trend = self.decomposition(x)

            # permute to (batch, in_dim, in_len) and apply linear layer on last dimension
            seasonal_output = self.linear_seasonal(res.permute(0, 2, 1))
            trend_output = self.linear_trend(trend.permute(0, 2, 1))

            x = seasonal_output + trend_output

            # extract nr_params
            x = x.view(batch, self.output_dim, self.output_chunk_length, self.nr_params)

            # permute back to (batch, out_len, out_dim, nr_params)
            x = x.permute(0, 2, 1, 3)
        
        else:
            res, trend = self.decomposition(x)

            # (in_len * in_dim) => (out_len * out_dim)
            seasonal_output = self.linear_seasonal(res.view(batch, -1))
            trend_output = self.linear_trend(trend.view(batch, -1))
            seasonal_output = seasonal_output.view(
                batch, self.output_chunk_length, self.output_dim
            )
            trend_output = trend_output.view(
                batch, self.output_chunk_length, self.output_dim
            )
            
            x = seasonal_output + trend_output

            if self.future_cov_dim != 0:
                x_future = torch.nn.functional.pad(
                    input=x_future,
                    pad=(0, 0, 0, self.output_chunk_length - x_future.shape[1])
                )

                fut_cov_output = self.linear_fut_cov(x_future)
                x = x + fut_cov_output.view(
                    batch, self.output_chunk_length, self.output_dim
                )
            
            if self.static_cov_dim != 0:
                static_cov_output = self.linear_static_cov(x_static.reshape(batch, -1))
                x = x + static_cov_output.view(
                    batch, self.output_chunk_length, self.output_dim
                )
            
            #
            x = torch.squeeze(x, dim=2)   # (배치 사이즈, 예측 기간, 일별 판매량) 반환
        return x