from src.common.model import get_best_tf_weights


def test_get_best_tf_weights():
    paths = [
        "data/runs/rnn/arch1-fold1/train1/models/epoch_01__val_accuracy_0.7044.keras",
        "data/runs/rnn/arch1-fold1/train1/models/epoch_02__val_accuracy_0.8995.keras",
        "data/runs/rnn/arch1-fold1/train1/models/epoch_03__val_accuracy_0.9578.keras",
        "data/runs/rnn/arch1-fold1/train1/models/epoch_05__val_accuracy_0.9597.keras",
        "data/runs/rnn/arch1-fold1/train1/models/epoch_07__val_accuracy_0.9607.keras",
        "data/runs/rnn/arch1-fold1/train1/models/epoch_09__val_accuracy_0.9640.keras",
        "data/runs/rnn/arch1-fold1/train1/models/epoch_10__val_accuracy_0.9675.keras",
        "data/runs/rnn/arch1-fold1/train2/models/epoch_01__val_accuracy_0.9557.keras",
        "data/runs/rnn/arch1-fold1/train2/models/epoch_02__val_accuracy_0.9607.keras",
        "data/runs/rnn/arch1-fold1/train2/models/epoch_04__val_accuracy_0.9613.keras",
        "data/runs/rnn/arch1-fold1/train2/models/epoch_05__val_accuracy_0.9626.keras",
        "data/runs/rnn/arch1-fold1/train2/models/epoch_06__val_accuracy_0.9748.keras",
        "data/runs/rnn/arch1-fold1/train3/models/epoch_02__val_accuracy_0.9692.keras",
        "data/runs/rnn/arch1-fold1/train3/models/epoch_01__val_accuracy_0.9650.keras",
    ]
    expected = (
        "data/runs/rnn/arch1-fold1/train2/models/epoch_06__val_accuracy_0.9748.keras"
    )

    actual = get_best_tf_weights(paths)

    assert actual == expected


def test_get_best_balanced_tf_weights():
    paths = [
        "data/runs/rnn/arch1-fold1/train1/models/epoch_01__val_loss_0.7044.keras",
        "data/runs/rnn/arch1-fold1/train1/models/epoch_02__val_loss_0.8995.keras",
        "data/runs/rnn/arch1-fold1/train1/models/epoch_03__val_loss_0.9578.keras",
        "data/runs/rnn/arch1-fold1/train1/models/epoch_05__val_loss_0.9597.keras",
        "data/runs/rnn/arch1-fold1/train1/models/epoch_07__val_loss_0.9607.keras",
        "data/runs/rnn/arch1-fold1/train1/models/epoch_09__val_loss_0.9640.keras",
        "data/runs/rnn/arch1-fold1/train1/models/epoch_10__val_loss_0.9675.keras",
        "data/runs/rnn/arch1-fold1/train2/models/epoch_01__val_loss_0.9557.keras",
        "data/runs/rnn/arch1-fold1/train2/models/epoch_02__val_loss_0.9607.keras",
        "data/runs/rnn/arch1-fold1/train2/models/epoch_04__val_loss_0.9613.keras",
        "data/runs/rnn/arch1-fold1/train2/models/epoch_05__val_loss_0.9626.keras",
        "data/runs/rnn/arch1-fold1/train2/models/epoch_06__val_loss_0.9748.keras",
        "data/runs/rnn/arch1-fold1/train3/models/epoch_02__val_loss_0.9692.keras",
        "data/runs/rnn/arch1-fold1/train3/models/epoch_01__val_loss_0.9650.keras",
    ]
    expected = "data/runs/rnn/arch1-fold1/train1/models/epoch_01__val_loss_0.7044.keras"

    actual = get_best_tf_weights(paths)

    assert actual == expected
