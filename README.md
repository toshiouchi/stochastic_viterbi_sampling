# Stochastic Viterbi Sampling Python Code

In the field of image captioning, we are conducting reinforcement learning of PPO that takes CRF into account. In reinforcement learning, it is common to sample captions probabilistically in a multinomial manner to calculate the sampled log probability and reward. It seems that inferred captions used to evaluate generated captions are usually generated greedily using argmax (max). The Viterbi algorithm seems greedy, and while it can be used for inferred captions, it cannot be used for probabilistically sampled captions that calculate rewards. Therefore, I wondered if it would be possible to devise a probabilistically sampled Viterbi algorithm. I would like to publish the results. I leave it up to the readers to decide whether it is theoretically correct.

As for the reinforcement learning, I haven't yet completed the training that I can publish online, so I'll only publish the code for stochastic Viterbi sampling here. Even if reinforcement learning doesn't work, I think I might be able to improve the Viterbi decoding version.

# Code to generate one sample

The function input emissions assumes that bert's last_hidden_state is set to ( bsz, seq_len, vocab_size ) using nn.LayerNorm and nn.Linear .

```python
class StochasticViterbiSample(nn.Module):
    def __init__(self, num_embedding, low_rank=32, beam_size=256, temp = 1.0):
        super().__init__()

        self.E1 = nn.Embedding(num_embedding, low_rank)
        self.E2 = nn.Embedding(num_embedding, low_rank)

        self.rank = low_rank
        self.beam = beam_size
        self.temp = temp

    def _compute_stochastic_viterbi_sample(self, emissions, beam=None):

        eps = 1e-8
        device = emissions.device
        
        beam = beam if beam is not None else self.beam
        
        beam_emission_scores, beam_targets = torch.topk( emissions, beam, 2)        
        
        batch_size, seq_len = beam_emission_scores.size()[:2]

        beam_transition_score1 = self.E1(beam_targets[:, :-1])  # B x (T-1) x K x D
        beam_transition_score2 = self.E2(beam_targets[:, 1:])   # B x (T-1) x K x D
        beam_transition_matrix = torch.bmm(
            beam_transition_score1.view(-1, beam, self.rank),
            beam_transition_score2.view(-1, beam, self.rank).transpose(1, 2))
        beam_transition_matrix = beam_transition_matrix.view(batch_size, -1, beam, beam) # bsz, seq_len, beam, beam

        traj_tokens, traj_scores = [], []
        finalized_tokens, finalized_scores = [], []

        # compute the normalizer in the log-space
        score = beam_emission_scores[:, 0]  # B x K
        
        for i in range(1, seq_len):
            traj_scores.append(score)
            _score = score[:, :, None] + beam_transition_matrix[:, i-1] # bsz, beam, beam

            # greedy selection
            #_score, _index = _score.max(dim=1) # bsz, beam     bsz, beam 

            # multinomial selection
            B, C, W = _score.shape
            flat_score = _score.permute(0, 2, 1).reshape(-1, C)
            probs = F.softmax(flat_score / self.temp, dim=-1)
            _index_flat = torch.multinomial(probs, num_samples=1)
            _score_flat = torch.gather(flat_score, -1, _index_flat)
            _index = _index_flat.view(B, W)
            _score = _score_flat.view(B, W)

            _score = _score + beam_emission_scores[:, i] # bsz, beam
            
            #if masks is not None:
            #    score = torch.where(masks[:, i: i+1], _score, score)
            #    index = torch.where(masks[:, i: i+1], _index, dummy)
            #else:
            score, index = _score, _index
            traj_tokens.append(index)
        
        all_scores = traj_scores
        all_scores.append( score )
        all_scores = torch.stack( all_scores, dim = 0 ).transpose( 0, 1 ).to(device)
        beam_probs = F.softmax( all_scores / self.temp, dim = 2 )

        # now running the back-tracing and find the best
        best_score, best_index = score.max(dim=1)
        finalized_tokens.append(best_index[:, None])
        finalized_scores.append(best_score[:, None])

        for idx, scs in zip(reversed(traj_tokens), reversed(traj_scores)):
            previous_index = finalized_tokens[-1]
            finalized_tokens.append(idx.gather(1, previous_index))
            finalized_scores.append(scs.gather(1, previous_index))

        finalized_tokens.reverse()
        sampled_beam_idx = torch.cat(finalized_tokens, 1)
        finalized_tokens = beam_targets.gather(2, sampled_beam_idx[:,:,None])[:, :, 0]

        finalized_scores.reverse()
        finalized_scores = torch.cat(finalized_scores, 1)
        finalized_scores[:, 1:] = finalized_scores[:, 1:] - finalized_scores[:, :-1]

        return beam_probs, sampled_beam_idx.unsqueeze(-1), finalized_tokens
```
```python
eps = 1e-8
tmp = torch.clamp( beam_probs, eps )
beam_log_probs = torch.log( tmp )
sample_log_probs = torch.gather( beam_log_probs, -1, sampled_beam_idx ).squeeze( -1 )
```

# Code to generate multiple samples

Since multiple samplings are required for the baseline calculation of GRPO, I came up with this code.

```python
class StochasticViterbiSamples(nn.Module):
    def __init__(self, num_embedding, low_rank=32, beam_size=256, temp = 1.0, num_samples = 8 ):
        super().__init__()

        self.E1 = nn.Embedding(num_embedding, low_rank)
        self.E2 = nn.Embedding(num_embedding, low_rank)

        self.rank = low_rank
        self.beam = beam_size
        self.temp = temp
        self.num_samples = num_samples

    def _compute_grpo_samples(self, emissions, masks=None, beam=None):

        eps = 1e-8
        device = emissions.device
        
        beam = beam if beam is not None else self.beam
        
        beam_emission_scores, beam_targets = torch.topk( emissions, beam, 2)        
        
        batch_size, seq_len = beam_emission_scores.size()[:2]

        beam_transition_score1 = self.E1(beam_targets[:, :-1])  # B x (T-1) x K x D
        beam_transition_score2 = self.E2(beam_targets[:, 1:])   # B x (T-1) x K x D
        beam_transition_matrix = torch.bmm(
            beam_transition_score1.view(-1, beam, self.rank),
            beam_transition_score2.view(-1, beam, self.rank).transpose(1, 2))
        beam_transition_matrix = beam_transition_matrix.view(batch_size, -1, beam, beam) # bsz, seq_len, beam, beam

        traj_tokens, traj_scores = [], []
        finalized_tokens, finalized_scores = [], []

        # compute the normalizer in the log-space
        score = beam_emission_scores[:, 0, :,  None].expand( -1, -1, self.num_samples )  # B x K, N
        
        for i in range(1, seq_len):
            traj_scores.append(score)
            _score = score[:, :, None] + beam_transition_matrix[:, i-1, :, :, None] # bsz, beam, beam, 1

            # greedy selection
            #_score, _index = _score.max(dim=1) # bsz, beam     bsz, beam 

            # multinomial selection
            B, C, W, _ = _score.shape
            N = self.num_samples
            flat_score = _score.permute(0, 2, 3, 1).reshape(-1, C) # b * W * N, C
            probs = F.softmax(flat_score / self.temp, dim=-1)
            _index_flat = torch.multinomial(probs, num_samples=1)
            _score_flat = torch.gather(flat_score, -1, _index_flat)
            _index = _index_flat.view(B, W, N)
            _score = _score_flat.view(B, W, N)

            _score = _score + beam_emission_scores[:, i, :, None] # bsz, beam
            
            #if masks is not None:
            #    score = torch.where(masks[:, i: i+1], _score, score)
            #    index = torch.where(masks[:, i: i+1], _index, dummy)
            #else:
            score, index = _score, _index # bsz, N
            traj_tokens.append(index) 
        
        all_scores = traj_scores
        all_scores.append( score )
        all_scores = torch.stack( all_scores, dim = 0 ).transpose( 0, 1 ).to(device) #bsz, seq_len, beam, N
        beam_probs = F.softmax( all_scores / self.temp, dim = 2 ) #bsz, seq_len, beam, N
        
        # now running the back-tracing and find the best
        best_score, best_index = score.max(dim=1) # max( bsz, beam ), bsz, N
        finalized_tokens.append(best_index[:, None, :]) #bsz,1, N
        finalized_scores.append(best_score[:, None, :]) #bsz,1, N

        for idx, scs in zip(reversed(traj_tokens), reversed(traj_scores)): # each of seq_len -1, bsz, beam, N 
            previous_index = finalized_tokens[-1]
            finalized_tokens.append(idx.gather(1, previous_index))
            finalized_scores.append(scs.gather(1, previous_index))

        finalized_tokens.reverse() # seq_len, bsz, N
        sampled_beam_idx = torch.cat(finalized_tokens, 1) # seq_len, bsz, N
        finalized_tokens = beam_targets.gather(2, sampled_beam_idx)
        
        finalized_scores.reverse()
        finalized_scores = torch.cat(finalized_scores, 1)
        finalized_scores[:, 1:] = finalized_scores[:, 1:] - finalized_scores[:, :-1]
        
        return beam_probs, sampled_beam_idx, finalized_tokens
```

I hope this helps. Thank you.
