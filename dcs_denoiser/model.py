import torch
import torch.nn as nn
import torch.nn.functional as F


class Generalised_LSTMDenoiser(nn.Module):
    """Bidirectional-LSTM denoiser for variable-length DCS correlation curves.

    Maps a noisy field/intensity correlation curve to a clean one. The model is
    geometry-agnostic: the lag-time (tau) grid is passed in as an input channel
    alongside the signal, so one network handles curves from different
    geometries and tau ranges.

    Most users should not call this class directly -- ``dcs_denoiser.denoise``
    builds the two input channels and undoes the ``g - 1`` shift for you.

    Input
    -----
    x : (B, L, 2) float tensor
        Channel 0: normalised tau grid (see ``dcs_denoiser.normalise_taus``).
        Channel 1: noisy signal, mean-subtracted to ``g - 1``.
        L is the padded sequence length; shorter curves are zero-padded to L.
    lengths : (B,) int64 tensor
        Number of valid (unpadded) points in each curve. Used to pack the
        sequence so the LSTM never sees padding.

    Output
    ------
    (B, 1, L) float tensor
        Predicted clean signal as ``g - 1``. The final softplus keeps it
        non-negative, which is the correct floor once the target is ``g - 1``.
        The channel dimension is in the middle to match the masked loss
        functions used in training. Padded positions hold a non-zero value
        (softplus of the FC bias) and must be ignored.

    Parameters
    ----------
    input_size : int, default 2
        Number of input channels (tau + signal).
    hidden_size : int, default 64
        LSTM hidden units per direction.
    num_layers : int, default 4
        Stacked LSTM layers. ``dropout`` is applied between layers only.
    dropout : float, default 0.2
        Inter-layer dropout; ignored when ``num_layers == 1``.
    bidirectional : bool, default True
        If True the LSTM runs both directions and the FC head input width is
        ``hidden_size * 2``.
    """

    def __init__(self, input_size=2, hidden_size=64, num_layers=4, dropout=0.2, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, 1)

    def forward(self, x, lengths):
        # Pack so the LSTM skips padded steps. lengths must be on CPU for
        # pack_padded_sequence; enforce_sorted=False lets the batch stay in
        # dataset order.
        lengths_cpu = lengths.cpu()
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        lstm_out_packed, _ = self.lstm(x_packed)

        # Unpack back to the fixed padded length L so each output aligns with the
        # target and mask.
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out_packed, batch_first=True, total_length=x.size(1)
        )

        # Per-timestep linear head -> (B, L, 1).
        denoised = self.fc(lstm_out)

        # softplus keeps the prediction >= 0, the correct floor for a ``g - 1``
        # target.
        denoised = F.softplus(denoised)

        # (B, L, 1) -> (B, 1, L): the masked loss functions expect the channel
        # dimension in the middle.
        return denoised.transpose(1, 2)
