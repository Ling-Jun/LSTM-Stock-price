from model_build import build


def train(X, y, batch_size, epoch):
    regressor = build(X, y, batch_size, epoch)
    history = regressor.fit(X, y, epochs=epoch, batch_size=batch_size)
    print(history)
