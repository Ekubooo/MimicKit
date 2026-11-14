"""SMP with train-mode restoration before the caller resets after evaluation."""
from learning.smp_agent import SMPAgent


class SMPGetupAgent(SMPAgent):
    def _load_params(self, config):
        super()._load_params(config)
        # Scope absolute minibatches to Getup; other PPO tasks retain their
        # existing interpretation (batch size as a multiple of environment count).
        n = self.get_num_envs()
        for name in ("actor", "critic"):
            key = name + "_batch_samples"
            if key in config:
                count = config[key]
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError(key + " must be a positive integer.")
                count = min(count, n * self._steps_per_iter)
                setattr(self, "_" + name + "_batch_size", count / n)

    def test_model(self, num_episodes):
        previous = self._mode
        try:
            return super().test_model(num_episodes)
        finally:
            # BaseAgent.train_model resets immediately after test_model returns.
            # Ensure that reset uses the 90/10 training distribution, not starts.
            self.set_mode(previous)

    def _output_train_model(self, iter, out_model_file, int_out_dir):
        super()._output_train_model(iter, out_model_file, int_out_dir)
        self._env.clear_train_diagnostics()
