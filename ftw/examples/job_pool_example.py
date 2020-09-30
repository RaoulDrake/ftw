from ftw.agents.tf.ftw.job_pool import FTWJobPool

from ftw.environments import pong_duel
from concurrent import futures
import multiprocessing


def main():
    env_fn = lambda: pong_duel.PongDuelPbtEnv()

    pool = FTWJobPool(
        sample_environment_fn=env_fn,
        arenas=2, learners=4,
        pixel_observations=False,
        min_steps=10, t=1
    )
    with futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Start the jobs and mark each future with its job index.
        jobs_future_to_idx = {
            executor.submit(
                lambda: job.run()): i
            for i, job in enumerate(pool.get_jobs())}

    job_is_done = [False for _ in jobs_future_to_idx]
    for future in futures.as_completed(jobs_future_to_idx):
        idx = jobs_future_to_idx[future]
        try:
            job_is_done[idx] = future.result()
        except Exception as exc:
            print(f'Learner Loop {idx} threw an exception: {exc}')


if __name__ == '__main__':
    main()
