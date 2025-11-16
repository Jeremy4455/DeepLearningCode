import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        手动实现Transformer中的前馈神经网络
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        """
        super(FeedForwardNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    input_dim = 512
    hidden_dim = 2048
    output_dim = 512

    ffn = FeedForwardNetwork(input_dim, hidden_dim, output_dim)
    x = torch.randn(2, 10, input_dim)

    out = ffn(x)
    print(out.shape)