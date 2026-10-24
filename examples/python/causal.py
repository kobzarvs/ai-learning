# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Увидеть разницу между наблюдаемой связью и причинным эффектом вмешательства.
weather = torch.randn(10000)
ice_cream = weather+0.2*torch.randn(10000)
swimming = 2*weather+0.2*torch.randn(10000)
correlation = torch.corrcoef(torch.stack([ice_cream, swimming]))[0,1]
# В нашей причинной схеме купание зависит от погоды, не от мороженого.
intervention = torch.full_like(ice_cream, 100.)
swimming_after = 2*weather+0.2*torch.randn(10000)
print('Корреляция:', correlation.item())
print('Изменение среднего купания при вмешательстве:', (swimming_after-swimming).mean().item())
assert intervention.mean() > ice_cream.mean()
