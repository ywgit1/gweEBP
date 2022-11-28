import os
import torch
import sys
import xfr
from xfr import xfr_root
from xfr.models import whitebox_clrp as whitebox 
from xfr.models import resnet
import warnings

def create_wbnet(net_name, device):
    """ Helper function for included networks.

        The best ebp_subtree_mode can depend on the network. If set to None,
        it should be set to the best value.
    """
    # python3 pytorch code
    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available()
            else "cpu")

    if net_name == 'resnetv6_pytorch':

        resnet_path = os.path.join(xfr_root,
                                   'models/resnet101_l2_d512_twocrop.pth')
        model = resnet.resnet101v6(resnet_path, device)

        model.to(device)
        wbnet = whitebox.WhiteboxSTResnet(
            model,
        )
        net = whitebox.Whitebox(wbnet).to(device)

        net.match_threshold = 0.9636
        net.platts_scaling = 15.05

        return net

    elif net_name == 'resnetv4_pytorch':

        resnet_path = os.path.join(xfr_root,
                                   'models/resnet101v4_28NOV17_train.pth')
        model = resnet.resnet101v6(resnet_path, device)

        model.to(device)
        wbnet = whitebox.WhiteboxSTResnet(
            model,
        )
        wb = whitebox.Whitebox(wbnet).to(device)

        # From Caffe ResNetv4:
        # wb.match_threshold = 0.9252
        # wb.platts_scaling = 17.71
        wb.match_threshold = 0.9722
        wb.platts_scaling = 16.61
        return wb

    elif net_name == 'vggface2_resnet50':

        sys.path.append(os.path.join(
            xfr_root, 'models/resnet50_128_pytorch'))
        import resnet50_128
        param_path = os.path.join(
            xfr_root, 'models/resnet50_128_pytorch/resnet50_128.pth')

        net = resnet50_128.resnet50_128(param_path)
        net.to(device)
        wb = whitebox.Whitebox(whitebox.Whitebox_resnet50_128(net)).to(device)

        wb.match_threshold = 0.896200
        wb.platts_scaling = 15.921608

        return wb
    
    elif net_name == 'vgg16':
        from xfr.models import vgg_tri_2_vis
        param_path = os.path.join(
            xfr_root, 'models/vgg_tri_2_Mon_11Oct2021_221223_epoch30.pth')

        net = vgg_tri_2_vis.get_model(param_path)
        net.to(device)
        wb = whitebox.Whitebox(whitebox.WhiteboxVGG16(net), device=device).to(device)

        wb.match_threshold = 0.141500 # @ FPR=0.04950495049504951 & TPR=0.6532258064516129
        wb.platts_scaling = 4.186850

        return wb        

    elif net_name == 'CUHK-vgg16':
        from xfr.models import vgg_tri_2_vis
        param_path = os.path.join(
            xfr_root, 'models/vgg_tri_2_vis_202111301013_epoch29.pth')

        net = vgg_tri_2_vis.get_model(param_path)
        net.to(device)
        wb = whitebox.Whitebox(whitebox.WhiteboxVGG16(net)).to(device)

        wb.match_threshold = 0.06 # @ FPR=0.15
        wb.platts_scaling = -2.042301

        return wb

    elif net_name == 'lcnn9':
        param_path = os.path.join(
            xfr_root, 'models/lcnn9_tri_Wed_06Oct2021_173415_epoch30.pth')
        from xfr.models import lcnn9_tri
        net = lcnn9_tri.get_model(param_path)
        net.eval()
        net.to(device)
        wb = whitebox.Whitebox(whitebox.WhiteboxLightCNN9(net)).to(device)

        wb.match_threshold = 0.245
        wb.platts_scaling = 7.349339

        return wb

    elif net_name == 'CUHK-lcnn9':
        param_path = os.path.join(
            xfr_root, 'models/lcnn9_tri_bestacc.pth')
        from xfr.models import lcnn9_tri
        net = lcnn9_tri.get_model(param_path)
        net.eval()
        net.to(device)
        wb = whitebox.Whitebox(whitebox.WhiteboxLightCNN9(net)).to(device)

        wb.match_threshold = 0.645
        wb.platts_scaling = -2.512507

        return wb
        
    elif net_name == 'lightcnn':
        if not os.path.exists(os.path.join(
            xfr_root, 'models/LightCNN_29Layers_V2_checkpoint.pth.tar')
        ):
            raise RuntimeError(
                'Missing light-CNN model file. Download file '
                '"LightCNN_29Layers_V2_checkpoint.pth.tar" from '
                'https://github.com/AlfredXiangWu/LightCNN and copy into '
                'explainable_face_recognition/models/LightCNN_29Layers_V2_checkpoint.pth.tar')
            return None
        net = xfr.models.lightcnn.LightCNN_29Layers_v2(num_classes=80013)
        statedict = xfr.models.lightcnn.Load_Checkpoint(os.path.join(
            xfr_root, 'models/LightCNN_29Layers_V2_checkpoint.pth.tar'))
        net.load_state_dict(statedict)
        net.to(device)
        wb = whitebox.Whitebox(whitebox.WhiteboxLightCNN(net)).to(device)
        wb.match_threshold = 0.829200
        wb.platts_scaling = 10.877741
        return wb

    else:
        raise NotImplementedError(
            'create_wbnet does not implemented network "%s"' %
            net_name
        )
