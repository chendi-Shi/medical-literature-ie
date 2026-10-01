import torch
from transformers import BertConfig,BertModel
from nlp_lab.ie_context import ContextExtractor

def test_context_pooling_masks_padding_overlap_and_preserves_direction():
    model=ContextExtractor(BertModel(BertConfig(vocab_size=20,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16)),2,2)
    hidden=torch.arange(10,dtype=torch.float32)[None,:,None].expand(1,10,8).clone().requires_grad_()
    mask=torch.tensor([[False,True,True,True,True,True,True,False,False,False]])
    pairs=torch.tensor([[[1,2,5,6],[5,6,1,2],[2,4,3,5],[0,0,0,0]]])
    features=model.pair_features(hidden,pairs,mask)
    # Each expected mean is derived from the unpadded content positions.
    expected=torch.tensor([[1.5,5.5,3.5,0.,4.5,2.5,0.,0.],
                           [5.5,1.5,3.5,2.5,0.,0.,4.5,0.],
                           [3.,4.,0.,1.,5.5,1.5,6.,0.],
                           [0.,0.,0.,0.,2.5,0.,2.5,0.]])
    torch.testing.assert_close(features[0,:,:64].reshape(4,8,8)[:,:,0],expected)
    assert not torch.equal(features[0,0,-16:],features[0,1,-16:])
    features[:,:,:56].sum().backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad[:,7:].abs().sum()==0
