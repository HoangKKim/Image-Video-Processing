import torch


class ActivationPrime:
    # derivative of sigmoid - đạo hàm sigmoid
    @staticmethod
    def sigmoid_derivative(s):
        return s * (1 - s)

    # derivative of tanh - đạo hàm tanh
    @staticmethod
    def tanh_derivative(s):
        return 1 - torch.tanh(s) ** 2
