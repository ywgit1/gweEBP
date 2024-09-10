import sys
import numpy as np
import torch


def x_face(img1, img2):
    # import matplotlib.pyplot as plt
    import cv2
    from src import MapGenerator, colorblend
    # from demo import ArcFaceOctupletLoss
    import numpy as np
    # Instantiate the MapGenerator
    # MapGenerator = MapGenerator(inference_fn=ArcFaceOctupletLoss(batch_size=64))
    
    # # VGG16 [SWITCH MODEL] Need to change
    # from vis_paper.vgg_tri_2_vis import get_embedding as get_embedding_vgg
    # from vis_paper.vgg_tri_2_vis import get_model
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # model = get_model(
    #     '/media/kent/DISK2/E2ID/visualization_hiding_game/xfr/models/vgg_tri_2_Mon_11Oct2021_221223_epoch30.pth'
    # )
    # model.to(device)
    # MapGenerator = MapGenerator(inference_fn=get_embedding_vgg, model=model)  # VGG16
    
    # LCNN9 [SWITCH MODEL] Need to change
    from vis_paper.lcnn9_tri import get_embedding as get_embedding_lcnn9
    from vis_paper.lcnn9_tri import get_model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = get_model(
        "/media/kent/DISK2/E2ID/Vis_Orig_LCNN9/trained_models/"
        "lcnn9_log_results_Wed_06Oct2021_1535_Malta_SetA_anchor_sketch_type/ckt/lcnn9_tri_Wed_06Oct2021_173415_epoch30.pth"
    )
    model.to(device)
    MapGenerator = MapGenerator(inference_fn=get_embedding_lcnn9, model=model)  # LCNN9
    
    # # Load an example image pair
    # image_pair = (
    #     cv2.cvtColor(cv2.imread("./demo/img1.png"), cv2.COLOR_BGR2RGB).astype(np.float32) / 255,
    #     cv2.cvtColor(cv2.imread("./demo/img2.png"), cv2.COLOR_BGR2RGB).astype(np.float32) / 255,
    # )
    
    image_pair = (
        cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2RGB).astype(np.float32) / 255,
        cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2RGB).astype(np.float32) / 255,
    )
    
    # # Show example image pair
    # fig, ax = plt.subplots(1, 2)
    # fig.suptitle("Example Image Pair")
    # ax[0].imshow(image_pair[0]), ax[1].imshow(image_pair[1])
    # plt.show()
    
    # # Generate and visualize the explanation maps
    # fig, ax = plt.subplots(3, 2)
    # fig.suptitle("Explanation Maps for Method 1, 2 and 3")
    # map1_m1, map2_m1, cam1_m1, cam2_m1 = MapGenerator(*image_pair, method="1")  # using method 1 for explanation maps
    # ax[0, 0].imshow(map1_m1), ax[0, 1].imshow(map2_m1)
    # map1_m2, map2_m2, cam1_m2, cam2_m2 = MapGenerator(*image_pair, method="2")  # using method 2 for explanation maps
    # ax[1, 0].imshow(map1_m2), ax[1, 1].imshow(map2_m2)
    map1_m3, map2_m3, cam1_m3, cam2_m3 = MapGenerator(*image_pair, method="3")  # using method 3 for explanation maps
    # ax[2, 0].imshow(map1_m3), ax[2, 1].imshow(map2_m3)
    # plt.show()
    
    # # Blend the explanations maps with the original images and visualize
    # fig, ax = plt.subplots(3, 2)
    # fig.suptitle("Blended Explanation Maps for Method 1, 2 and 3")
    # # ax[0, 0].imshow(colorblend(image_pair[0], map1_m1)), ax[0, 1].imshow(colorblend(image_pair[1], map2_m1))
    # # ax[1, 0].imshow(colorblend(image_pair[0], map1_m2)), ax[1, 1].imshow(colorblend(image_pair[1], map2_m2))
    # ax[2, 0].imshow(colorblend(image_pair[0], map1_m3)), ax[2, 1].imshow(colorblend(image_pair[1], map2_m3))
    # plt.show()
    return cam1_m3
    
    
def normalize(deviations: np.array) -> np.array:
    """Normalize explanation maps from [-2, 2] first to [-1, 1] and then to the range [0, 1]

    :param deviations: The explanation maps
    :return: Normalized explanation maps
    """
    
    if np.max(np.abs(deviations)) != 0:
        deviations = deviations / np.max(np.abs(deviations))
    return (deviations + 1.0) / 2.0


if __name__ == '__main__':
    img1 = sys.argv[1]
    img2 = sys.argv[2]
    img3 = sys.argv[3]
    cam1 = x_face(img1, img2)
    cam2 = x_face(img1, img3)
    cam = cam1 - cam2
    
    cam = normalize(cam)
    np.save(f'/media/kent/DISK2/GitHub_Repos/x-face-verification/temp_res/res.npy', cam)  # Need to change
