use pyo3::{exceptions::PyValueError, prelude::*};

pub(crate) fn config(
    py: Python<'_>,
    k: usize,
    search_algorithm: &str,
    weighting: &str,
    tree_leaf_size: usize,
) -> PyResult<atlas_ml::KnnConfig> {
    let search_algorithm = match search_algorithm {
        "brute_force" => atlas_ml::KnnSearchAlgorithm::BruteForce,
        "kd_tree" => atlas_ml::KnnSearchAlgorithm::KdTree,
        "ball_tree" => atlas_ml::KnnSearchAlgorithm::BallTree,
        "auto" => atlas_ml::KnnSearchAlgorithm::Auto,
        _ => {
            return Err(PyValueError::new_err(
                "search_algorithm must be 'brute_force', 'kd_tree', 'ball_tree', or 'auto'",
            ));
        }
    };
    let weighting = match weighting {
        "uniform" => atlas_ml::KnnWeighting::Uniform,
        "distance" => atlas_ml::KnnWeighting::Distance,
        _ => return Err(PyValueError::new_err("weighting must be 'uniform' or 'distance'")),
    };

    atlas_ml::KnnConfig::new(k)
        .and_then(|config| config.with_search_algorithm(search_algorithm))
        .and_then(|config| config.with_tree_leaf_size(tree_leaf_size))
        .map(|config| config.with_weighting(weighting))
        .map_err(|error| crate::support::errors::ml(py, error))
}
