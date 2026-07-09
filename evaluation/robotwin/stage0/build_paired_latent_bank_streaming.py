"""Build a paired bank with training-aligned frame IDs and 1+4 VAE streaming."""

from common.observation_encoder_chunked import ChunkedRobotwinObservationEncoder

import build_paired_latent_bank_aligned as aligned


aligned.implementation.RobotwinObservationEncoder = ChunkedRobotwinObservationEncoder


if __name__ == "__main__":
    aligned.implementation.main()
