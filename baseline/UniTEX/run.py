from pipeline import CustomRGBTextureFullPipeline
import os
rgb_tfp = CustomRGBTextureFullPipeline(pretrain_models='./unitex_pretrained_models',
                                        super_resolutions=False,
                                        seed = 63)

test_image_path = "./reference_images/0543be3bb52696a6b71344104319446d7b4ca17941491b1bcfdf5e24e4ee652b.png"
test_mesh_path = "./test_set/0543be3bb52696a6b71344104319446d7b4ca17941491b1bcfdf5e24e4ee652b.glb"
save_root = './outputs/0543be3bb52696a6b71344104319446d7b4ca17941491b1bcfdf5e24e4ee652b'
os.makedirs(save_root, exist_ok=True)
rgb_tfp(save_root, test_image_path, test_mesh_path, clear_cache=False)