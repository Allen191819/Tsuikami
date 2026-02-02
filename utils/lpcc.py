import torch
import numpy as np
from scipy.linalg import toeplitz

class LPCC(torch.nn.Module):
    """Generate LPCC features for input to the speech pipeline."""

    def __init__(self, 
                 order=12, 
                 deltas=False, 
                 context=False, 
                 sample_rate=16000, 
                 win_length=25, 
                 hop_length=10):
        super().__init__()
        self.order = order  # LPC order (number of coefficients)
        self.deltas = deltas
        self.context = context
        self.sample_rate = sample_rate
        self.win_length = int(win_length * sample_rate / 1000)  # Convert ms to samples
        self.hop_length = int(hop_length * sample_rate / 1000)  # Convert ms to samples

    def _lpc_analysis(self, signal, order):
        """Compute LPC coefficients using autocorrelation method."""
        # Compute autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[autocorr.size // 2:]  # Only the second half (non-negative lags)
        
        # Set up the LPC coefficients (autocorrelation matrix and vector)
        R = toeplitz(autocorr[:order])  # Use scipy.linalg.toeplitz
        r = autocorr[1:order + 1]
        
        # Solve the normal equations to get the LPC coefficients
        lpc_coeffs = np.linalg.solve(R, r)
        return lpc_coeffs

    def _lpc_to_lpcc(self, lpc_coeffs):
        """Convert LPC coefficients to LPCC coefficients."""
        # Compute the LPCC from the LPC coefficients
        lpcc = np.zeros(self.order)
        for m in range(self.order):
            # Recursively compute the LPCCs
            lpcc[m] = -lpc_coeffs[m]
        return lpcc

    def _frame_signal(self, signal):
        """Frame the signal into overlapping windows."""
        num_frames = (len(signal) - self.win_length) // self.hop_length + 1
        frames = np.stack([signal[i * self.hop_length:i * self.hop_length + self.win_length] 
                           for i in range(num_frames)], axis=0)
        return frames

    def forward(self, wav):
        """Compute LPCC features for a batch of audio signals."""
        # Convert input waveform to a NumPy array
        device = wav.device
        wav = wav.cpu().numpy()
        
        lpcc_features = []
        
        for signal in wav:
            frames = self._frame_signal(signal)
            lpccs = []
            
            # For each frame, compute LPCC
            for frame in frames:
                # Compute LPC coefficients
                lpc_coeffs = self._lpc_analysis(frame, self.order)
                lpcc = self._lpc_to_lpcc(lpc_coeffs)
                lpccs.append(lpcc)
            
            lpccs = np.array(lpccs)
            lpcc_features.append(lpccs)
        
        lpcc_features = np.array(lpcc_features)  # Shape: (batch_size, num_frames, order)
        
        # Convert lpcc_features to torch tensor and move it to the same device as wav
        lpcc_features = torch.tensor(lpcc_features, dtype=torch.float32)
        lpcc_features = lpcc_features.to(device)  # Move to the same device as wav
        
        if self.deltas:
            # Compute delta and delta-delta features
            delta1 = self._compute_deltas(lpcc_features)
            delta2 = self._compute_deltas(delta1)
            lpcc_features = torch.cat([lpcc_features, delta1, delta2], dim=-1)
        
        if self.context:
            # Apply context window (e.g., concatenating left and right context frames)
            lpcc_features = self._apply_context_window(lpcc_features)
        
        return lpcc_features

    def _compute_deltas(self, features):
        """Compute delta features (first derivatives)."""
        deltas = features.new_zeros(features.shape)
        for t in range(1, features.shape[1] - 1):
            deltas[:, t] = features[:, t + 1] - features[:, t - 1]
        return deltas

    def _apply_context_window(self, features):
        """Apply a context window (concatenate left and right frames)."""
        # For simplicity, use a window size of 5 frames for both left and right context
        left_context = features[:, :-5, :]
        right_context = features[:, 5:, :]
        return torch.cat([left_context, right_context], dim=2)