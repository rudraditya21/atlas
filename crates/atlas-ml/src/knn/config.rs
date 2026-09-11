use crate::{AtlasMlError, AtlasMlResult};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KnnSearchAlgorithm {
    #[default]
    BruteForce,
    KdTree,
    BallTree,
    Auto,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KnnWeighting {
    #[default]
    Uniform,
    Distance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KnnConfig {
    k: usize,
    search_algorithm: KnnSearchAlgorithm,
    weighting: KnnWeighting,
}

impl KnnConfig {
    pub fn new(k: usize) -> AtlasMlResult<Self> {
        if k == 0 {
            return Err(AtlasMlError::InvalidArgument {
                op: "knn_config",
                reason: "k must be positive",
            });
        }

        Ok(Self {
            k,
            search_algorithm: KnnSearchAlgorithm::BruteForce,
            weighting: KnnWeighting::Uniform,
        })
    }

    pub const fn k(&self) -> usize {
        self.k
    }

    pub const fn search_algorithm(&self) -> KnnSearchAlgorithm {
        self.search_algorithm
    }

    pub fn with_search_algorithm(
        mut self,
        search_algorithm: KnnSearchAlgorithm,
    ) -> AtlasMlResult<Self> {
        validate_search_algorithm(search_algorithm)?;
        self.search_algorithm = search_algorithm;
        Ok(self)
    }

    pub const fn with_weighting(mut self, weighting: KnnWeighting) -> Self {
        self.weighting = weighting;
        self
    }

    pub const fn weighting(&self) -> KnnWeighting {
        self.weighting
    }

    pub fn validate(&self, training_samples: usize) -> AtlasMlResult<()> {
        validate_search_algorithm(self.search_algorithm)?;
        if self.k > training_samples {
            return Err(AtlasMlError::InvalidArgument {
                op: "knn_config",
                reason: "k must not exceed the number of training samples",
            });
        }

        Ok(())
    }
}

fn validate_search_algorithm(algorithm: KnnSearchAlgorithm) -> AtlasMlResult<()> {
    match algorithm {
        KnnSearchAlgorithm::BruteForce | KnnSearchAlgorithm::KdTree | KnnSearchAlgorithm::Auto => {
            Ok(())
        }
        KnnSearchAlgorithm::BallTree => Err(AtlasMlError::InvalidArgument {
            op: "knn_config",
            reason: "the requested search algorithm is not available",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{KnnConfig, KnnSearchAlgorithm, KnnWeighting};
    use crate::AtlasMlError;

    #[test]
    fn defaults_to_brute_force_search() {
        let config = KnnConfig::new(3).unwrap();

        assert_eq!(config.k(), 3);
        assert_eq!(config.search_algorithm(), KnnSearchAlgorithm::BruteForce);
        assert_eq!(config.weighting(), KnnWeighting::Uniform);
    }

    #[test]
    fn rejects_zero_neighbors() {
        assert_eq!(
            KnnConfig::new(0),
            Err(AtlasMlError::InvalidArgument { op: "knn_config", reason: "k must be positive" })
        );
    }

    #[test]
    fn rejects_neighbor_counts_larger_than_training_data() {
        let config = KnnConfig::new(3).unwrap();

        assert_eq!(
            config.validate(2),
            Err(AtlasMlError::InvalidArgument {
                op: "knn_config",
                reason: "k must not exceed the number of training samples",
            })
        );
    }

    #[test]
    fn accepts_available_algorithm_choices() {
        let brute_force = KnnConfig::new(3)
            .unwrap()
            .with_search_algorithm(KnnSearchAlgorithm::BruteForce)
            .unwrap();
        let kd_tree =
            KnnConfig::new(3).unwrap().with_search_algorithm(KnnSearchAlgorithm::KdTree).unwrap();
        let automatic =
            KnnConfig::new(3).unwrap().with_search_algorithm(KnnSearchAlgorithm::Auto).unwrap();

        assert_eq!(brute_force.search_algorithm(), KnnSearchAlgorithm::BruteForce);
        assert_eq!(kd_tree.search_algorithm(), KnnSearchAlgorithm::KdTree);
        assert_eq!(automatic.search_algorithm(), KnnSearchAlgorithm::Auto);
    }

    #[test]
    fn rejects_unavailable_algorithm_choices() {
        assert_eq!(
            KnnConfig::new(3).unwrap().with_search_algorithm(KnnSearchAlgorithm::BallTree),
            Err(AtlasMlError::InvalidArgument {
                op: "knn_config",
                reason: "the requested search algorithm is not available",
            })
        );
    }
}
