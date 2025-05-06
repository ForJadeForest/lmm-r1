import torch

def experience_filter(experience_maker,experiences):
    args = experience_maker.strategy.args
    rewards = torch.cat([experience.info["reward"] for experience in experiences])
    rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
    lower_bound = 0
    upper_bound = 1
    invalid_group = (rewards == lower_bound).all(dim=-1,keepdim=True) | (rewards == upper_bound).all(dim=-1,keepdim=True) # [group_size, 1]
    invalid_group = invalid_group.repeat(1,args.n_samples_per_prompt) # [group_size, n_samples_per_prompt]
    invalid_group = invalid_group.flatten().reshape(len(experiences),-1) # [experiences_size, micro_rollout_batch_size]
    assert invalid_group.size(1) == args.micro_rollout_batch_size
    assert args.n_samples_per_prompt % invalid_group.size(1) == 0, f"Group size should be a divisor of micro_rollout_batch_size, but got {args.n_samples_per_prompt} and {invalid_group.size(1)}"
    assert len(invalid_group) == len(experiences), f"Invalid group shape: {invalid_group.shape}, experiences shape: {len(experiences)}, rewards shape: {rewards.shape}"
    experiences = [experiences[i] for i in range(len(experiences)) if not invalid_group[i][0]]
    return experiences