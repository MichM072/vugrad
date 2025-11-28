import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.visualization import plot_param_importances
import optuna.visualization as vis

def plot_optuna_results(study, variant):
    best_trial = study.best_trial
    print(f"Best trial loss: {best_trial.value}")
    print(f"Best parameters: {best_trial.params}")
    optuna.visualization.plot_param_importances(study)
    opt_history_plot = vis.plot_optimization_history(study)
    opt_param_importances_plot = vis.plot_param_importances(study)

    # Save all plots
    opt_history_plot.write_image(f"optuna_history_{variant}.png")
    opt_param_importances_plot.write_image(f"optuna_param_importances_{variant}.png")

    opt_history_plot.show()
    opt_param_importances_plot.show()

    # Best trial loss per epoch
    loss_best_trial = best_trial.intermediate_values.values()
    x = np.arange(1, len(loss_best_trial)+1)
    plt.plot(x, loss_best_trial)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'q12_{variant}_best_trial.png')
    # plt.show()
    return loss_best_trial

if __name__ == "__main__":
    variants = [0, 1, 2, 3]
    all_losses = []
    for variant in variants:
        study = optuna.load_study(study_name=f"cifar_net_q12_{variant}",
                                  storage="sqlite:///optuna.db")
        loss = plot_optuna_results(study, variant=0)
        all_losses.append(loss)

    fig, ax = plt.subplots()
    for i, loss in enumerate(all_losses):
        x = np.arange(1, len(loss) + 1)
        ax.plot(x, loss, label=f"Variant {i}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    plt.savefig('q12_best_trial_all.png')
    plt.show()