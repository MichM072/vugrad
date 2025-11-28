import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.visualization import plot_param_importances
import optuna.visualization as vis

def plot_optuna_results(study):
    best_trial = study.best_trial
    print(f"Best trial loss: {best_trial.value}")
    print(f"Best parameters: {best_trial.params}")
    optuna.visualization.plot_param_importances(study)
    opt_history_plot = vis.plot_optimization_history(study)
    opt_param_importances_plot = vis.plot_param_importances(study)

    # Save all plots
    opt_history_plot.write_image("optuna_history.png")
    opt_param_importances_plot.write_image("optuna_param_importances.png")

    opt_history_plot.show()
    opt_param_importances_plot.show()

    # Best trial loss per epoch
    loss_best_trial = best_trial.intermediate_values.values()
    x = np.arange(1, len(loss_best_trial)+1)
    plt.plot(x, loss_best_trial)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('q11_best_trial.png')
    plt.show()

if __name__ == "__main__":
    study = optuna.load_study(study_name="cifar_net_q11",
                              storage="sqlite:///optuna.db")
    plot_optuna_results(study)