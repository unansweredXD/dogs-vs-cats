import wandb


def init_wandb(learning_rate=0.02, epochs=10, batch_size=32):
    wandb.init(
        project='demo',

        config={
            'learning_rate': learning_rate,
            'architecture': 'CNN',
            'dataset': 'dogsvscats',
            'batch_size': batch_size,
            'epochs': epochs,
        }
    )
