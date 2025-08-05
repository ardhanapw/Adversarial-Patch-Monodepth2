import torch
from networks.resnet_encoder import ResnetEncoder
from networks.depth_decoder import DepthDecoder

def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            return func(*f_args, **f_kwargs)
    return wrapper

class AdversarialModels:
    def __init__(self, args):
        self.args = args
        if 'monodepth2' in self.args.model:
            self.distill = DistillModel(require_grad=True).cuda().eval()
            self.fix_distill = DistillModel(require_grad=False).cuda().eval()
            print("=> Using MonoDepth2 (ResNet encoder + depth decoder)")

    def load_weights(self):
        if hasattr(self, 'distill'):
            print("=> Loading MonoDepth2 (.pth) pretrained weights")
            enc_w = torch.load(self.args.encoder_path, map_location='cpu')
            enc_w = {k: v for k, v in enc_w.items()
                     if k in self.distill.encoder.state_dict()}
            self.distill.encoder.load_state_dict(enc_w)
            self.fix_distill.encoder.load_state_dict(enc_w)

            dec_w = torch.load(self.args.decoder_path, map_location='cpu')
            self.distill.decoder.load_state_dict(dec_w)
            self.fix_distill.decoder.load_state_dict(dec_w)
            print("=> Weights loaded successfully")
            
    @make_nograd_func
    def get_original_disp(self, sample):
        add_sample = {}
        if 'monodepth2' in self.args.model:
            distill_disp = self.fix_distill(sample['left'])
            add_sample.update({"original_distill_disp": distill_disp.detach()})
        return add_sample

    """
    @make_nograd_func
    def get_original_disp(self, sample):
        if 'monodepth2' in self.args.model:
            disp = self.fix_distill(sample['left'])
            return {"original_distill_disp": disp.detach()}
        return {}
    """

class DistillModel(torch.nn.Module):
    def __init__(self, require_grad=True):
        super().__init__()
        self.encoder = ResnetEncoder(num_layers=18, pretrained=True)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc,
                                    scales=range(4))
        if not require_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, input_img):
        f = self.encoder(input_img)
        #print(f"encoder result: {len(f)}")
        out = self.decoder(f)
        #print(f"decoder result: {len(out)}")
        return out[("disp", 0)]