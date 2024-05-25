import numpy
from torch.utils.data import DataLoader, Subset


from algos.base_class import BaseServer


class IsolatedServer(BaseServer):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.model_save_path = "{}/saved_models/node_{}.pt".format(self.config["results_path"],
                                                                   self.node_id)
        self.set_training_data(config)

    def set_training_data(self, config):
        train_dset = self.dset_obj.train_dset
        test_dset = self.dset_obj.test_dset
        samples_per_client = config["samples_per_client"]
        batch_size = config["batch_size"]
        client_idx = self.node_id
        indices = numpy.random.permutation(len(train_dset))
        dset = Subset(train_dset, indices[client_idx*samples_per_client:(client_idx+1)*samples_per_client])
        self.dloader = DataLoader(dset, batch_size=batch_size*len(self.device_ids), shuffle=True)

    def test(self) -> float:
        """
        Test the model on the server
        """
        test_loss, acc = self.model_utils.test(self.model,
                                               self._test_loader,
                                               self.loss_fn,
                                               self.device)
        # TODO save the model if the accuracy is better than the best accuracy so far
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return acc

    def train(self):
        """
        Runs the whole training procedure
        """
        self.log_utils.log_console("Starting training epoch")
        loss = self.model_utils.train(self.model,
                                      self.optim,
                                      self.dloader,
                                      self.loss_fn,
                                      self.device)
        self.log_utils.log_console("Training epoch done")
        return loss

    def run_protocol(self):
        self.log_utils.log_console("Starting isolated client training")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for round in range(start_epochs, total_epochs):
            self.log_utils.log_console("Starting round {}".format(round))
            
            loss, tr_acc = self.train()
            self.log_utils.log_tb(f"train_loss/clients", loss, round)
            self.log_utils.log_console("round: {} train_loss:{:.4f}".format(
                round, loss
            ))
            
            acc = self.test()
            self.log_utils.log_tb(f"test_acc/clients", acc, round)
            self.log_utils.log_console("round: {} test_acc:{:.4f}".format(
                round, acc
            ))
            self.log_utils.log_console("Round {} done".format(round))
        self.log_utils.log_console("Isolated client training over")