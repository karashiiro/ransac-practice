#![no_std]

extern crate alloc;
use alloc::vec;
use rand_core::RngCore;

pub struct Data<const P: usize> {
    n_datapoints: usize,
    data: [[f64; P]],
}

pub trait Estimator<const P: usize, const S: usize> {
    fn fit_hypothesis(self: &Self, hypothesis: &[f64; S]);

    fn residual(self: &Self, datapoint: &[f64; P]) -> f64;
}

pub fn ransac<R, E, const P: usize, const S: usize>(
    mut rng: R,
    estimator: E,
    dataset: &Data<P>,
    inlier_threshold: f64,
    outlier_ratio: f64,
    confidence: f64,
) -> [f64; P]
where
    R: RngCore,
    E: Estimator<P, S>,
{
    let n_trials = libm::ceil(
        libm::log10(1.0 - confidence) / libm::log10(1.0 - libm::pow(1.0 - outlier_ratio, S as f64)),
    ) as usize;

    let mut best_hypothesis_score: u64 = 0; // The number of points within the inlier threshold
    let mut best_hypothesis: [f64; P] = vec![0.0; P]; // The parameters of the best-fit model
    for _ in 0..n_trials {
        // Select S random points to fit the model to.
        let mut current_hypothesis: &[f64; S] = vec![0.0; S];
        for x in current_hypothesis.iter_mut() {
            *x = rng.next_u64().wrapping_mul(dataset.n_datapoints as u64) as f64;
        }

        // Score the current hypothesis
        estimator.fit_hypothesis(current_hypothesis);
        let hypothesis_score: u64 = dataset
            .data
            .iter()
            .map(|data| {
                // Calculate the residual of the data point. If it is less
                // than the inlier threshold, add 1 to the hypothesis score.
                let residual = libm::fabs(estimator.residual(data));
                if residual < inlier_threshold {
                    return 1;
                } else {
                    return 0;
                }
            })
            .sum();

        // Replace the best hypothesis if needed
        if hypothesis_score > best_hypothesis_score {
            best_hypothesis_score = hypothesis_score;

            for (x0, x1) in best_hypothesis.iter_mut().zip(current_hypothesis.iter()) {
                *x0 = *x1;
            }
        }
    }

    return best_hypothesis;
}
