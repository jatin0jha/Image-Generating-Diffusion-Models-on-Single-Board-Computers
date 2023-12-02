# Image-Generating-Diffusion-Models-on-Single-Board-Computers

![Project Logo/Image]

## Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## About

<Short description of your project, including its purpose and goals.>

## Features

- Efficient deployment of image generating diffusion models on single board computers.
- Utilizes a lightweight diffusion model and an efficient inference engine.
- Evaluation on various image datasets showcasing the generation of high-quality images on single board computers.

## Installation

To run this project, follow these steps:

1. Check the GPU:

    ```bash
    !nvidia-smi
    ```

2. Install dependencies:

    ```bash
    !git clone --recursive https://github.com/crowsonkb/v-diffusion-pytorch
    %pip install git+https://github.com/crowsonkb/k-diffusion
    ```

3. Download the diffusion model:
   
    ```bash
    !mkdir v-diffusion-pytorch/checkpoints
    !curl -L --http1.1 https://the-eye.eu/public/AI/models/v-diffusion/cc12m_1_cfg.pth > v-diffusion pytorch/checkpoints/cc12m_1_cfg.pth
    ```

4. Imports:
   
    ```bash
    import gc
    import math
    import sys
    
    import clip
    from IPython import display
    import k_diffusion as K
    import torch
    from torch import nn
    from torchvision import utils
    from torchvision.transforms import functional as TF
    from tqdm.notebook import trange, tqdm
    
    sys.path.append('/content/v-diffusion-pytorch')
    
    from diffusion import get_model
    ```

5. Load the models:
   
    ```bash
    inner_model = get_model('cc12m_1_cfg')()
    _, side_y, side_x = inner_model.shape
    inner_model.load_state_dict(torch.load('v-diffusion-pytorch/checkpoints/cc12m_1_cfg.pth', map_location='cpu'))
    inner_model = inner_model.half().cuda().eval().requires_grad_(False)
    model = K.external.VDenoiser(inner_model)
    clip_model = clip.load(inner_model.clip_model, jit=False, device='cpu')[0]
    ```

6. The diffusion model:
   
    ```bash
    target_embed = clip_model.encode_text(clip.tokenize(prompt)).float().cuda()


    class CFGDenoiser(nn.Module):
        def __init__(self, model, cond_scale):
            super().__init__()
            self.inner_model = model
            self.cond_scale = cond_scale
    
    def forward(self, x, sigma, clip_embed):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        clip_embed_in = torch.cat([torch.zeros_like(clip_embed), clip_embed])
        uncond, cond = self.inner_model(x_in, sigma_in, clip_embed=clip_embed_in).chunk(2)
        return uncond + (cond - uncond) * self.cond_scale
    
    
    def callback(info):
        if info['i'] % display_every == 0:
            nrow = math.ceil(info['denoised'].shape[0] ** 0.5)
            grid = utils.make_grid(info['denoised'], nrow, padding=0)
            tqdm.write(f'Step {info["i"]} of {steps}, sigma {info["sigma"]:g}:')
            display.display(K.utils.to_pil_image(grid))
            tqdm.write(f'')
    
    
    def run():
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(seed)
        sigmas = K.sampling.get_sigmas_karras(steps, 1e-2, 160, device='cuda')
        x = torch.randn([n_images, 3, side_y, side_x], device='cuda') * sigmas[0]
        model_wrap = CFGDenoiser(torch.cuda.amp.autocast()(model), weight)
        extra_args = {'clip_embed': target_embed.repeat([n_images, 1])}
        outs = K.sampling.sample_lms(model_wrap, x, sigmas, extra_args=extra_args, callback=callback)
        tqdm.write('Done!')
        for i, out in enumerate(outs):
            filename = f'out_{i}.png'
            K.utils.to_pil_image(out).save(filename)
            display.display(display.Image(filename))
    
    
    run()
    ```


## Usage

paste the given code on google colab paste all the given code snipits make sure ur machine has a gpu then run all the snippets in given order

-Check the GPU
-Install dependencies
-Download the diffusion model
-Imports
-Load the models
-

## Results

<Highlight key results or findings from your evaluations. Include visual outputs or links to demo images if applicable.>

## Contributing

If you'd like to contribute to this project, please follow these guidelines:

- Fork the repository
- Create a new branch for your feature: `git checkout -b feature-name`
- Make your changes and commit them: `git commit -m 'Add feature'`
- Push to the branch: `git push origin feature-name`
- Submit a pull request

## License

This project is licensed under the [License Name] - see the [LICENSE.md](LICENSE.md) file for details.

