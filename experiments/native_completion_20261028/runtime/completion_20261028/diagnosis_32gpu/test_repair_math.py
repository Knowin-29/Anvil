import torch
from repair_train import token_unlikelihood,cycle_endpoint,Pool
# Negative-token loss and gradients must agree with direct -log(1-p).
torch.manual_seed(5)
head=torch.nn.Linear(4,7,bias=False).double();head.requires_grad_(False)
a=torch.randn(1,8,4,dtype=torch.double,requires_grad=True)
b=a.detach().clone().requires_grad_();ids=torch.tensor([[1,2,3,4,5,6,1,2]]);pos=[2,3,5,7]
actual=token_unlikelihood(a,head,ids,pos)
logits=head(b[0,torch.tensor(pos)-1]).float();p=logits.softmax(-1).gather(1,ids[0,pos,None]).squeeze(1)
expected=-torch.log1p(-p).mean();actual.backward();expected.backward()
assert torch.allclose(actual.float(),expected.float(),atol=1e-6)
assert torch.allclose(a.grad,b.grad,atol=1e-6)
zero=token_unlikelihood(a,head,ids,[]);assert zero.item()==0 and zero.requires_grad
assert cycle_endpoint(list(range(800)))==(None,None)
assert cycle_endpoint([1,2,3,4]*192)[1]['period_tokens']==4
for family in 'AG':
 pool=Pool(family)
 for micro in range(20):
  for rank in range(2):
   row=pool.sample(micro,rank,2)
   assert row['gsa_meta']['split']=='train' and row['gsa_meta']['binding']
print('PASS: stable unlikelihood values and gradients; connected zero; conservative cycle detection; source-bound QA and task sampler')
