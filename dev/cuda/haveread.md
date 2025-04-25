adam
encoder-f-b

gelu -f -b

global norm 其实就是对一个整个的数据块进行归约

layernorm -f

crossentropy_forward crossentropy_softmax_backward 实现不难，主要是没理解交叉熵损失的作用

fused_residual_forward.cu

residual_forward

permute