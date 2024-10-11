import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import QNN

class FConv2D_3x3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        """
        In the forward pass, we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # Perform the convolution with quantized weights
        input_clipped = toch.clamp(input, 0, 255)
        conv_result = F.conv2d(input_clipped, QNN.quantize(weight))

        # Add bias if present
        if bias is not None:
            reshapedBias = bias.reshape(1, bias.shape[0], 1, 1).repeat(conv_result.shape[0], 1, conv_result.shape[2], conv_result.shape[3])
            conv_result += reshapedBias

        # Apply floor operation to simulate quantization
        result = torch.floor(conv_result)

        # Clip the result between 0 and 255 to simulate restricted resolution
        result_clipped = torch.clamp(result, 0, 255)

        # Save tensors for backward pass
        ctx.save_for_backward(input, weight, bias, result_clipped)

        return result_clipped

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """
        In the backward pass, we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input, weight, and bias.
        """
        input, weight, bias, result_clipped = ctx.saved_tensors

        # Compute gradients with respect to the input, weight, and bias
        grad_input = grad_weight = grad_bias = None

        # Create a mask to ignore gradients where the forward result was clipped
        clip_mask = (result_clipped > 0) & (result_clipped < 255)

        # Apply the mask to the gradient coming from the output
        grad_output = grad_output * clip_mask

        if ctx.needs_input_grad[0]:
            grad_input = F.conv_transpose2d(grad_output, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = F.conv2d(input.transpose(0, 1), grad_output.transpose(0, 1)).transpose(0, 1)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        return grad_input, grad_weight, grad_bias

class Conv2D_3x3(nn.Module):
    def __init__(self, inChannels, outChannels, dtype=torch.float32, bias=True):
        super(Conv2D_3x3, self).__init__()

        self.inChannels = inChannels
        self.outChannels = outChannels
        self.dtype = dtype

        self.weight = torch.nn.Parameter(torch.empty((outChannels, inChannels, 3, 3), dtype=dtype))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(outChannels, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters (from ConvNd.reset_parameters)
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return FConv2D_3x3.apply(input, self.weight, self.bias)
