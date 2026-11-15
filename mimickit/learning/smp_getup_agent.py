"""SMP with train-mode restoration before the caller resets after evaluation."""
import torch
from engines.engine import ControlMode
from learning.smp_agent import SMPAgent


class SMPGetupAgent(SMPAgent):
    def _build_model(self, config):
        super()._build_model(config)
        init_pose = config.get("actor_init_pose", "default")
        if init_pose == "default":
            return
        if init_pose != "standing":
            raise ValueError("actor_init_pose must be default or standing.")
        if self._env._engine.get_control_mode() not in (ControlMode.pos, ControlMode.pd_explicit):
            raise ValueError("Standing actor initialization requires position control.")
        # Only initialize the fresh actor's output bias. Keep the absolute PD
        # action mapping and checkpoint format unchanged; load() overrides this.
        target = self._env._init_dof_pos
        if not torch.isfinite(target).all() or not torch.allclose(self._env._clip_action(target), target):
            raise ValueError("Standing initialization must be a finite, reachable PD target.")
        bias = self._model._action_dist._mean_net.bias
        with torch.no_grad():
            bias.copy_(self._a_norm.normalize(target).to(bias))

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
            # Ensure that reset uses the configured training mixture, not starts.
            self.set_mode(previous)

    def _record_data_pre_step(self, obs, info, action, action_info):
        super()._record_data_pre_step(obs, info, action, action_info)
        # Record the episode label at collection time: slots may reset during
        # a rollout, so the environment's final mask cannot label the buffer.
        self._exp_buffer.record('getup_standing', info['getup_init_standing'])

    def _build_train_data(self):
        task = self._exp_buffer.get_data_flat('reward').clone()
        info = super()._build_train_data()
        combined = self._exp_buffer.get_data_flat('reward')
        weighted_task = self._task_reward_weight * task
        weighted_smp = combined - weighted_task
        standing = self._exp_buffer.get_data_flat('getup_standing').bool()
        advantage = self._exp_buffer.get_data_flat('adv')
        for label, mask in [('standing', standing), ('nonstanding', ~standing)]:
            # Nonstanding also covers fallen/augmented starts in other Getup
            # configs; only the near-standing curriculum makes it reference-only.
            prefix = 'getup_rollout_' + label + '_'
            info[prefix + 'samples'] = mask.sum()
            for name, value in [('task', weighted_task), ('smp', weighted_smp),
                                ('adv', advantage), ('positive_adv', (advantage > 0).float())]:
                info[prefix + name] = value[mask].mean() if mask.any() else value.new_zeros(())
        return info

    def _output_train_model(self, iter, out_model_file, int_out_dir):
        super()._output_train_model(iter, out_model_file, int_out_dir)
        self._env.clear_train_diagnostics()
