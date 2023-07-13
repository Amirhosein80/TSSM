class EarlyStopping:
    """
    Early Stopping for Overfit :)
    """

    def __init__(self, tolerance: int = 5, min_delta: float = 0.5) -> None:
        """
        :param tolerance: number of epochs for wait :)
        :param min_delta: allowed difference between train loss & valid loss
        """
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        """
        check overfit :)
        :param train_loss: training loss
        :param validation_loss: validation loss
        :return:
        """
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
