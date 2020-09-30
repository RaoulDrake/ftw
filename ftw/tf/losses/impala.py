"""Losses for IMPALA, UNREAL, FTW and similar agents.

Taken from the original IMPALA repository at https://github.com/deepmind/scalable_agent."""
import tensorflow as tf


def compute_baseline_loss(advantages):
    # Loss for the baseline, summed over the time dimension.
    # Multiply by 0.5 to match the standard update rule:
    # d(loss) / d(baseline) = advantage
    return .5 * tf.reduce_sum(tf.square(advantages))


def compute_entropy_loss(logits):
    policy = tf.nn.softmax(logits)
    log_policy = tf.nn.log_softmax(logits)
    entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
    return -tf.reduce_sum(entropy_per_timestep)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions, logits=logits)
    advantages = tf.stop_gradient(advantages)
    policy_gradient_loss_per_timestep = cross_entropy * advantages
    return tf.reduce_sum(policy_gradient_loss_per_timestep)
