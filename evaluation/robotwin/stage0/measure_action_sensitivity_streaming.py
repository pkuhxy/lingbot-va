"""Action sensitivity entry point using chunked policy observation encoding."""

from common.policy_harness_streaming import StreamingDeterministicPolicyHarness

import measure_action_sensitivity as implementation


implementation.DeterministicPolicyHarness = StreamingDeterministicPolicyHarness


if __name__ == "__main__":
    implementation.main()
