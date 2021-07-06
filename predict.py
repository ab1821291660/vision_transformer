import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from vit_model import vit_base_patch16_224_in21k as create_model
####
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
def main(img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img = Image.open(img_path)##===================================
    plt.imshow(img)##===================================
    img = data_transform(img)
    ####b1-c3-224-224
    img = torch.unsqueeze(img, dim=0)
    ####
    # read class_indict
    json_path = './class_indices.json'##===================================
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    ####===================================================================================================================================
    # create model
    model = create_model(num_classes=2, has_logits=False).to(device)
    # load model weights
    model_weight_path = "./weights/model-1.pth"##===================================
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    ####===================================================================================================================================
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)####class: dog   prob: 0.586
    plt.show()
if __name__ == '__main__':
    # img_path = "../tulip.jpg"
    img_path = "./tt_input/cat.9.jpg"
    main(img_path)




