# Eliminating Gradient Conflict in Reference-based Line-Art Colorization

> The official code of **ECCV2022** paper: [**Eliminating Gradient Conflict in Reference-based Line-Art Colorization**](https://arxiv.org/abs/2207.06095)
> <br>Zekun Li, Zhengyang Geng, Zhao Kang, Wenyu Chen, Yibo Yang

We propose a new attention module called **Stop-Gradient Attention**. Our main idea is detaching the gradient when backpropagating the attention map.

The module is illustrated as follows:

<img src="./images/SGA.png" height="450">

The core code is showed as follow:

```python3
# input:
# X: feature maps -> tensor(b, wh, c)
# Y: feature maps -> tensor(b, wh, c)
# output:
# Z: feature maps -> tensor(b, wh, c)
# other objects:
# Wq, Wv: embedding matrix -> nn.Linear(c,c)
# A: attention map -> tensor(b, wh, wh)
# leaky_relu: leaky relu activation function
with torch.no_grad():
    A = X.bmm(Y.permute(0, 2, 1))
    A = softmax(A, dim=-1)
    A = normalize(A, p=1, dim=-2)
X = leaky_relu(Wq(X))
Y = leaky_relu(Wv(Y))
Z = torch.bmm(A,Y) + X
```

# bibtex
```
@inproceedings{li2022eliminating,
  title={Eliminating Gradient Conflict in Reference-based Line-Art Colorization},
  author={Li, Zekun and Geng, Zhengyang and Kang, Zhao and Chen, Wenyu and Yang, Yibo},
  booktitle={European Conference on Computer Vision},
  pages={579--596},
  year={2022},
  organization={Springer}
}
```
