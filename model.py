from model_starter import image_loader
from model_starter import run_style_transfer
def save_output(output):
    filename = 'output'
    img = transforms.ToPILImage(mode='RGB')(output.squeeze(0))
    img.save(f'{filename}.png')


def style_tranfer(cnn, imsize, device, style_img_path, content_img_path):
    style_img = image_loader(style_img_path, imsize, device)
    content_img = image_loader(content_img_path, imsize, device)
    input_img = content_img.clone()
    output = run_style_transfer(cnn, device, content_img, style_img, input_img)
    save_output(output)