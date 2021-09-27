
class HyperSearch:
    # optimize for several metrics, not eval_loss
    direction = "maximize"
    backend = "optuna"

    @staticmethod
    def get_search_space(trial):
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 1e-2, log=True
            ),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
            "seed": trial.suggest_int("seed", 1, 42),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [4, 8, 16, 32]
            ),
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps", [1, 2]
            ),
        }
