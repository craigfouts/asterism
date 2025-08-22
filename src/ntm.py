"""
Craig Fouts (craig.fouts@uu.igp.se)
"""

from torch import nn
from torch.utils.data import DataLoader
from base import OPTIM, HotTopic, Encoder, MLP

class GSM(HotTopic, nn.Module):
    def __init__(self, max_topics=100, *, channels=(64, 32), kld_scale=.1, optim='adam', desc='GSM', random_state=None):
        super().__init__(desc, random_state)

        self.max_topics = max_topics
        self.channels = channels
        self.kld_scale = kld_scale
        self.optim = optim

        self._n_steps = 200
    
    def _build(self, X, learning_rate=1e-2, batch_size=32, shuffle=True):
        in_channels, self._batch_size = X.shape[-1], batch_size
        self._loader = DataLoader(X, self._batch_size, shuffle)
        self._encoder = Encoder(in_channels, *self.channels)
        self._g_model = MLP(self.channels[-1], self.max_topics, final_act='softmax', dim=-1)
        self._decoder = MLP(self.max_topics, in_channels)
        self._optim = OPTIM[self.optim](self.parameters(), lr=learning_rate)
        self.train()

        return self
    
    def _step(self):
        loss = 0.

        for x in self._loader:
            z, kl = self._encoder(x, return_kld=True)
            x_ = self._decoder(self._g_model(z))
            x_loss = (x_ - x).square().sum().sqrt() + self.kld_scale*kl
            x_loss.backward()
            loss += x_loss.item()/self._batch_size

        self._optim.step()
        self._optim.zero_grad()

        return loss
    
    def _predict(self, X, eval=True):
        if eval:
            self.eval()

        topics = (X@self._decoder[0][0].weight.detach()).argmax(-1)

        return topics
    
    def forward(self, X, eval=True):
        topics = self._predict(X, eval)

        return topics
