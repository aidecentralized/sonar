# /////////////// Gradient Inversion Helpers ///////////////

import inversefed
from collections import OrderedDict
import matplotlib.pyplot as plt

# based on InvertingGradients code by Jonas Geiping
# code found in https://github.com/JonasGeiping/invertinggradients/tree/1157b61c6704df42c497ab9eb074c75da5204334

def compute_param_delta(param_s, param_t, basic_params):
    """
    Generates the input value for reconstruction
    Assumes param_s and param_t are from the same client.

    basic_params: list of names present in model params
    """
    return [(param_t[name] - param_s[name]).detach() for name in basic_params if name in param_s and name in param_t]

def reconstruct_gradient(param_diff, target_labels, lr, local_steps, model, client_id=0):
    """
    Reconstructs the gradient following the Geiping InvertingGradients technique
    """


    dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]

    config = dict(signed=True,
                boxed=True,
                cost_fn='sim',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='adam',
                restarts=1,
                max_iterations=8_000,
                total_variation=1e-6,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
    
    rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), local_steps, lr, config,
                                                use_updates=True, num_images=len(target_labels))
    
    output, stats = rec_machine.reconstruct(param_diff, target_labels, img_shape=(3, 32, 32))

    grid_plot(output, target_labels, ds, dm, save_path=f"gias_output_client_{client_id}.png")    
    return output, stats

def grid_plot(tensor, labels, ds, dm, save_path=None):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)

    fig, axes = plt.subplots(1, 10, figsize=(24, 24))
    for im, l, ax in zip(tensor, labels, axes.flatten()):
        ax.imshow(im.permute(1, 2, 0).cpu())
        ax.set_title(l)
        ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # Save the figure if save_path is provided
    # plt.show()  # Show the plot after saving

def gia_main(param_s, param_t, model):
    """
    Main function for Gradient Inversion Attack
    """
    params = model.parameters().keys()
    param_diff = compute_param_delta(param_s, param_t, params)
    output, stats = reconstruct_gradient(param_diff, 3e-4 , 1, model)
    return output, stats