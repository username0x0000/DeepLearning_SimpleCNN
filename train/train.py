import config as cfg


def train_main():
    dataloader = get_dataloader(cfg.DATA, 'train')
    model = get_model(cfg.MODEL)
    loss = get_loss(cfg.LOSS)
    optimizer = get_optimizer(cfg.OPTIMIZER)

    trainer = Trainer(model, loss, optimizer, cfg.TRAIN)
    trainer.train(dataloader)

    dataloader = get_dataloader(cfg.DATA, 'validation')
    validater = Validater(model)
    validater.eval(dataloader)


def get_loss(cfg):
    if cfg['type'] == 'CrossEntropy':
        loss = torch.cross_entropy()

    return loss


def get_optimizer():
    pass
