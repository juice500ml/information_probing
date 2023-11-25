# This helper code summarizes the test set statistics
# from multiple torch lightning-generated tensorboard directories.

from pathlib import Path
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


if __name__ == "__main__":
    search_str = input("Experiment name to search: ")

    accs, losses = [], []

    for layer_index in (0, 6, 12, 18, 23):
        paths = list(Path("exps/").glob(f"*{search_str}-l{layer_index}*/lightning_logs/version_*"))
        assert len(paths), paths

        event_acc = EventAccumulator(str(paths[0]))
        event_acc.Reload()
        loss, acc = event_acc.Scalars("test/loss"), event_acc.Scalars("test/acc")
        assert len(loss) == 1
        assert len(acc) == 1
        losses.append(loss[0].value)
        accs.append(acc[0].value)

    print('"acc":', accs)
    print('"loss":', losses)
