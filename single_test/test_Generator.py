from Network import *

d_model = 756
vocab = 10000
generator = Generator(d_model, vocab)
# 假设 x 是一个 [batch_size, d_model] 的 tensor
hidden_state = torch.randn(32, d_model)  # [batch_size, d_model] => [32, 756]
# Forward pass
output = generator(hidden_state)  # After Linear + log_softmax
print(output.shape)  # Should be [32, vocab] => [32, 10000]
