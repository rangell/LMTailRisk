import numpy as np
import torch
from typing import Optional
from transformers import DynamicCache


def update_cache_after_resampling(
    past_key_values: DynamicCache,
    indices: torch.Tensor,
    model_config,
):
    indices = indices.cpu()
    for layer in past_key_values.layers:
        # NOTE: in general this is super dangerous, but ok in this situation
        layer.keys = layer.keys[indices]
        layer.values = layer.values[indices]
    return past_key_values


class FKSteering:
    def __init__(
        self,
        device,
        r_fn,
        potential_type,
        max_seq_len,
        num_particles,
        resample_start,
        resample_end,
        resample_interval,
        lmbda,
        use_smc,
        adaptive_resampling: Optional[bool] = True,
        adaptive_resampling_threshold: Optional[float] = 0.5,
        smc_verbose: Optional[bool] = False,
        importance_resampling_at_last_step: Optional[bool] = False,
        use_importance_weights_in_resampling: Optional[bool] = False,
    ):
        self.use_smc = use_smc
        self.importance_resampling_at_last_step = importance_resampling_at_last_step

        self.device = device

        self.r_fn = r_fn
        self.lmbda = lmbda
        self.smc_verbose = smc_verbose

        self.num_particles = num_particles
        self.adaptive_resampling = adaptive_resampling
        self.adaptive_resampling_threshold = adaptive_resampling_threshold

        self.resample_start = resample_start
        self.resample_end = resample_end
        self.resample_interval = resample_interval
        self.max_seq_len = max_seq_len
        self.use_importance_weights_in_resampling = use_importance_weights_in_resampling

        if self.use_smc:
            self.resampling_arr = torch.arange(
                resample_start, resample_end + 1, resample_interval
            )
            self.resampling_arr = self.resampling_arr - 1
            self.resampling_arr = self.resampling_arr.to(device)

        else:
            self.resampling_arr = torch.arange(max_seq_len, device=device)

        # self.resampling_arr = torch.cat(
        #     [self.resampling_arr, torch.tensor([max_seq_len])]
        # )

        if smc_verbose:
            print("Resampling steps:", self.resampling_arr.tolist())
            print("Not resampling at last step:")

        self.potential_type = potential_type
        assert potential_type in ["r_fn", "max", "diff", "bon"], potential_type

        self.arr_r_values = [torch.zeros(num_particles, device=device)]
        self.arr_potential_values = []

        # partial p(x_t, ...,x_{t+r} | x_{1:t}) / q(x_t, ...,x_{t+r} | x_{1:t}
        self.accum_importance_weights = torch.ones(num_particles, device=device)

        # complete p(x_t, ...,x_{t+r} | x_{1:t}) / q(x_t, ...,x_{t+r} | x_{1:t}
        # self.arr_importance_weights = torch.ones(num_particles, device=device)

    def resampling_fn(self, step_idx, potential_values, importance_weights):
        """
        Resampling function that returns indices based on the importance weights.

        Input:
        w: unnormalized importance weights, shape (N,)

        Output:
        indices: indices for resampling, shape (N,)
        """
        num_particles = potential_values.shape[0]

        w = potential_values
        if self.use_importance_weights_in_resampling:
            if self.smc_verbose:
                print("using importance weights for resampling")
            w = potential_values * importance_weights.view(num_particles)
        # w = potential_values  # * importance_weights.view(num_particles)
        assert w.shape == (num_particles,), (
            w.shape,
            importance_weights.shape,
            potential_values.shape,
        )

        # Normalize the weights
        normalized_w = w / torch.sum(w)
        ess = 1.0 / torch.sum(torch.pow(normalized_w, 2)).item()

        if (
            step_idx == self.max_seq_len - 1
            and self.importance_resampling_at_last_step is False
        ):
            print(
                "Not resampling at last step, using uniform potential values. ESS / k:",
                ess / num_particles,
            )
            return np.arange(num_particles), potential_values

        if self.adaptive_resampling:
            if ess < self.adaptive_resampling_threshold * num_particles:
                indices = self.stratified_resampling_fn(normalized_w)

                if self.smc_verbose:
                    print(
                        "Resampling triggered due to low ESS / k:", ess / num_particles
                    )
                    print(
                        "Unique resampling indices:",
                        len(np.unique(indices, return_counts=False)),
                    )

            else:
                indices = np.arange(num_particles)

                # if adaptive resampling is not triggered, set potential values to uniform distribution
                if self.smc_verbose:
                    print(
                        "Adaptive resampling not triggered, using uniform potential values. ESS:",
                        ess / num_particles,
                    )
                potential_values = torch.ones_like(potential_values)
        else:
            # If adaptive resampling is not used, always resample
            # This is a fallback to ensure resampling occurs
            indices = self.stratified_resampling_fn(normalized_w)
            if self.smc_verbose:
                print("Adaptive resampling is disabled, resampling will always occur.")
                print(
                    "Unique resampling indices:",
                    len(np.unique(indices, return_counts=False)),
                )

        return indices, potential_values

    def stratified_resampling_fn(self, w):
        """
        Stratified resampling of particles according to their weights.

        Args:
            weights: 1D array-like, normalized weights (sum to 1).

        Returns:
            indices: array of indices of resampled particles (ints, same length as weights)
        """
        w = w.cpu().numpy()
        N = len(w)
        positions = (np.random.rand(N) + np.arange(N)) / N
        cumsum = np.cumsum(w)
        indices = np.zeros(N, dtype=int)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumsum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return indices

    def compute_potential(self, step_idx, sequence, rs_candidates):
        if self.potential_type == "r_fn":
            raise NotImplementedError("r_fn potential type not implemented")
        elif self.potential_type == "max":
            raise NotImplementedError("Max potential type not implemented")
        elif self.potential_type == "diff":
            rs_old = self.arr_r_values[-1]
            return torch.exp(self.lmbda * (rs_candidates - rs_old))

        elif self.potential_type == "bon":
            if step_idx == 0:
                return torch.ones_like(rs_candidates)
        else:
            raise ValueError(f"Unknown potential type: {self.potential_type}")

    def update_history(self, indices, arr_r_values, arr_potential_values):
        for past_idx in range(len(arr_r_values)):
            arr_r_values[past_idx] = arr_r_values[past_idx][indices]

        for past_idx in range(len(arr_potential_values)):
            arr_potential_values[past_idx] = arr_potential_values[past_idx][indices]

        return arr_r_values, arr_potential_values

    def __call__(self, step_idx, sequences, importance_weights, rs_candidates):
        # collect product of importance_weights
        self.accum_importance_weights = (
            self.accum_importance_weights * importance_weights.view(self.num_particles)
        )

        if step_idx not in self.resampling_arr or not self.use_smc:
            # If not resampling, just append one as potential values
            self.arr_potential_values.append(
                torch.ones(self.num_particles, device=self.device)
            )

            # If not resampling, just return the sequences and indices
            return torch.arange(self.num_particles, device=self.device)

        # rs_candidates = self.r_fn(sequences)

        # print(f"Sample {step_idx}, r_fn candidates shape: {rs_candidates}")
        if self.smc_verbose:
            print(
                f"step {step_idx}. r_fn candidates: {rs_candidates.mean().item()} +- {rs_candidates.std().item()}"
            )
        assert rs_candidates.shape == (self.num_particles,), rs_candidates.shape

        potential_values = self.compute_potential(step_idx, sequences, rs_candidates)
        assert potential_values.shape == (self.num_particles,), potential_values.shape

        indices, potential_values = self.resampling_fn(
            potential_values=potential_values,
            importance_weights=self.accum_importance_weights,
            step_idx=step_idx,
        )

        num_particles, seq_len = sequences.shape

        # Update r_values and potential_values to new indices
        arr_r_values, arr_potential_values = self.update_history(
            indices,
            arr_r_values=self.arr_r_values,
            arr_potential_values=self.arr_potential_values,
        )

        self.arr_r_values = arr_r_values
        self.arr_potential_values = arr_potential_values
        # self.arr_importance_weights = self.arr_importance_weights[indices]

        self.arr_r_values.append(rs_candidates[indices])
        self.arr_potential_values.append(potential_values[indices])

        # Update accumulated importance weights
        self.accum_importance_weights = torch.ones(
            self.num_particles, device=self.device
        )

        return torch.tensor(indices)

    def get_fk_quantities(self):
        assert self.potential_type == "diff"

        assert (
            self.potential_type == "diff"
        ), "FK estimate only available for 'diff' potential type"

        arr_potential_values = torch.stack(self.arr_potential_values, dim=1)

        assert arr_potential_values.shape == (
            self.num_particles,
            self.max_seq_len,
        ), arr_potential_values.shape

        normalization_constant = torch.exp(
            torch.sum(torch.log(arr_potential_values.mean(dim=0)))
        )

        Z = normalization_constant

        assert Z > 0, f"Z = {Z} must be positive for FK estimate"

        inv_potential = torch.exp(-torch.sum(torch.log(arr_potential_values), dim=1))
        assert inv_potential.shape == (self.num_particles,)

        normalized_potential_values = (
            arr_potential_values[:, -1] / arr_potential_values[:, -1].sum()
        )

        return dict(
            inv_potential=inv_potential.cpu().numpy(),
            Z=Z.cpu().numpy(),
            arr_potential_values=arr_potential_values.cpu().numpy(),
            normalized_potential_values=normalized_potential_values.cpu().numpy(),
        )

    def compute_fk_estimate(self, test_function_values, importance_weights):
        assert self.potential_type == "diff"

        assert (
            self.potential_type == "diff"
        ), "FK estimate only available for 'diff' potential type"

        arr_potential_values = torch.stack(self.arr_potential_values, dim=1)

        assert arr_potential_values.shape == (
            self.num_particles,
            self.max_seq_len,
        ), arr_potential_values.shape

        # product_of_potentials = torch.exp(
        #     torch.sum(torch.log(arr_potential_values), dim=1)
        # )
        # assert product_of_potentials.shape == (self.num_particles,), (
        #     product_of_potentials.shape,
        # )

        normalization_constant = torch.exp(
            torch.sum(torch.log(arr_potential_values.mean(dim=0)))
        )

        Z = normalization_constant

        assert Z > 0, f"Z = {Z} must be positive for FK estimate"

        inv_potential = torch.exp(-torch.sum(torch.log(arr_potential_values), dim=1))

        assert inv_potential.shape == (self.num_particles,)

        if self.use_importance_weights_in_resampling:
            importance_weights = torch.ones_like(importance_weights)
            print(
                "Ignoring importance weights in FK estimate since they were used in resampling"
            )
            print("Warning: TURNED OFFFFF" * 40)
            print("Warning: TURNED OFFFFF" * 40)
            pass

        imp_weighted_test_function_values = (
            test_function_values * importance_weights.view(self.num_particles)
        )
        if self.importance_resampling_at_last_step:
            # If importance resampling is used at the last step, we use the accumulated importance weights
            estimate = (
                Z * (inv_potential * imp_weighted_test_function_values).mean().item()
            )
        else:
            # import pdb; pdb.set_trace()
            normalized_potential_values = (
                arr_potential_values[:, -1] / arr_potential_values[:, -1].sum()
            )
            if self.smc_verbose:
                print(
                    "normalized_potential_values",
                    normalized_potential_values.mean().item(),
                    normalized_potential_values.std().item(),
                )

            estimate = (
                Z
                * (
                    inv_potential
                    * imp_weighted_test_function_values
                    * normalized_potential_values
                ).sum()
            )

        if self.smc_verbose:
            print(
                f"FK estimate: {estimate}, Z: {Z}, inv_potential: {inv_potential.mean().item()}"
            )
            print(
                "test_fn values mean",
                test_function_values.mean().item(),
                test_function_values.std().item(),
            )
            print("importance_weight_arr", importance_weights.mean().item())
            print("arr_potential_values", arr_potential_values.mean().item())
            print(
                "Normalized potential values", normalized_potential_values.sum().item()
            )

        return torch.tensor([estimate])
