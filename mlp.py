import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.7),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        # x = nn.Softmax(dim=1)(x)
        return x
    
if __name__ == "__main__":
    test_input = torch.rand((8, 100, 18))
    model = MLP(1800, 64, 2)
    output = model(test_input)
    print(output)