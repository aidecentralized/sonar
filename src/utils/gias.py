# /////////////// Gradient Inversion Helpers ///////////////

import inversefed
import matplotlib.pyplot as plt

import torch
import pickle

# based on InvertingGradients code by Jonas Geiping
# code found in https://github.com/JonasGeiping/invertinggradients/tree/1157b61c6704df42c497ab9eb074c75da5204334

def compute_param_delta(param_s, param_t, basic_params):
    """
    Generates the input value for reconstruction
    Assumes param_s and param_t are from the same client.

    basic_params: list of names present in model params
    """
    assert len(param_s) != 0 and len(param_t) != 0, "Empty parameters"
    return [(param_t[name].to("cuda") - param_s[name].to("cuda")).detach() for name in basic_params if name in param_s and name in param_t]

def reconstruct_gradient(param_diff, target_labels, target_images, lr, local_steps, model, client_id=0):
    """
    Reconstructs the gradient following the Geiping InvertingGradients technique
    """
    print("length of param diff: ", len(param_diff))
    with open(f"param_diff_{client_id}.pkl", "wb") as f:
        pickle.dump(param_diff, f)
    setup = inversefed.utils.system_startup()
    for p in range(len(param_diff)):
        param_diff[p] = param_diff[p].to(setup['device'])
    # param_diff = param_diff.to(setup['device'])
    target_labels = target_labels.to(setup['device'])
    target_images = target_images.to(setup['device'])

    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    dm = torch.as_tensor(mean, **setup)[:, None, None]
    ds = torch.as_tensor(std, **setup)[:, None, None]
    model = model.to(setup['device'])
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
    
    assert len(param_diff) == 38

    rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), local_steps, lr, config,
                                                use_updates=True, num_images=len(target_labels))
    
    output, stats = rec_machine.reconstruct(param_diff, target_labels, img_shape=(3, 32, 32))

    # compute reconstruction acccuracy
    test_mse = (output.detach() - target_images).pow(2).mean()
    feat_mse = (model(output.detach())- model(target_images)).pow(2).mean()  
    test_psnr = inversefed.metrics.psnr(output, target_images, factor=1/ds)
    print(f"Client {client_id} Test MSE: {test_mse:.2e}, Test PSNR: {test_psnr:.2f}, Feature MSE: {feat_mse:.2e}")

    grid_plot(output, target_labels, ds, dm, stats, test_mse, feat_mse, test_psnr, save_path=f"gias_output_client_{client_id}.png")    
    return output, stats

def grid_plot(tensor, labels, ds, dm, stats, test_mse, feat_mse, test_psnr, save_path=None):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)

    fig, axes = plt.subplots(1, 10, figsize=(24, 24))
    for im, l, ax in zip(tensor, labels, axes.flatten()):
        ax.imshow(im.permute(1, 2, 0).cpu())
        ax.set_title(l)
        ax.axis('off')
    plt.title(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
            f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |");
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # Save the figure if save_path is provided
    # plt.show()  # Show the plot after saving

def gia_main(param_s, param_t, base_params, model, target_labels, target_images, client_id):
    """
    Main function for Gradient Inversion Attack
    Returns results moved back to their original devices
    """
    # Store original devices
    model_device = next(model.parameters()).device
    target_labels_device = target_labels.device
    target_images_device = target_images.device
    
    # Store original parameter devices
    param_s_devices = {name: param_s[name].device for name in base_params if name in param_s}
    param_t_devices = {name: param_t[name].device for name in base_params if name in param_t}
    
    param_diff = compute_param_delta(param_s, param_t, base_params)

    # Check if all elements in para_diff are zero tensors
    if all((diff == 0).all() for diff in param_diff):
        print("Parameter differences contain only zeros for client ", client_id)
        return None  # or return an empty list, depending on your needs
    
    output, stats = reconstruct_gradient(param_diff, target_labels, target_images, 3e-4, 1, model, client_id)
    
    # Move output back to target_images device (since it's a reconstruction of the images)
    if output is not None:
        output = output.to(target_images_device)
    
    # Move model back to original device
    model.to(model_device)
    
    # Move parameters back to their original devices
    for name in base_params:
        if name in param_s:
            param_s[name] = param_s[name].to(param_s_devices[name])
        if name in param_t:
            param_t[name] = param_t[name].to(param_t_devices[name])
            
    # Move labels and images back to their original devices
    target_labels = target_labels.to(target_labels_device)
    target_images = target_images.to(target_images_device)
            
    return output, stats