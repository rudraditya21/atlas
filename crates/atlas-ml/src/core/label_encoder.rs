use std::collections::BTreeSet;

use crate::{AtlasMlError, AtlasMlResult};

const FIT_OP: &str = "label_encoder_fit";
const TRANSFORM_OP: &str = "label_encoder_transform";
const INVERSE_TRANSFORM_OP: &str = "label_encoder_inverse_transform";

/// A deterministic mapping between ordered labels and contiguous class indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelEncoder<L> {
    classes: Box<[L]>,
}

impl<L> LabelEncoder<L>
where
    L: Ord + Clone,
{
    /// Fits an encoder with classes sorted in ascending label order.
    pub fn fit(labels: &[L]) -> AtlasMlResult<Self> {
        if labels.is_empty() {
            return Err(AtlasMlError::EmptyInput { op: FIT_OP });
        }

        let classes: Box<[L]> =
            labels.iter().cloned().collect::<BTreeSet<_>>().into_iter().collect();

        Ok(Self { classes })
    }

    /// Returns the ascending labels corresponding to encoded indices.
    pub fn classes(&self) -> &[L] {
        &self.classes
    }

    /// Encodes labels into their contiguous class indices.
    pub fn transform(&self, labels: &[L]) -> AtlasMlResult<Vec<usize>> {
        labels
            .iter()
            .map(|label| {
                self.classes.binary_search(label).map_err(|_| AtlasMlError::InvalidArgument {
                    op: TRANSFORM_OP,
                    reason: "label was not observed during fitting",
                })
            })
            .collect()
    }

    /// Decodes contiguous class indices back into their original labels.
    pub fn inverse_transform(&self, indices: &[usize]) -> AtlasMlResult<Vec<L>> {
        indices
            .iter()
            .map(|&index| {
                self.classes.get(index).cloned().ok_or(AtlasMlError::InvalidArgument {
                    op: INVERSE_TRANSFORM_OP,
                    reason: "encoded label index is out of bounds",
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::LabelEncoder;
    use crate::AtlasMlError;

    #[test]
    fn encodes_sorted_unique_classes_and_duplicate_labels() {
        let encoder = LabelEncoder::fit(&["pear", "apple", "pear", "orange"]).unwrap();

        assert_eq!(encoder.classes(), &["apple", "orange", "pear"]);
        assert_eq!(encoder.transform(&["pear", "apple", "orange", "pear"]), Ok(vec![2, 0, 1, 2]));
    }

    #[test]
    fn rejects_empty_fitting_labels_and_unknown_values() {
        let encoder = LabelEncoder::fit(&["apple", "pear"]).unwrap();

        assert_eq!(
            LabelEncoder::<&str>::fit(&[]),
            Err(AtlasMlError::EmptyInput { op: "label_encoder_fit" })
        );
        assert_eq!(
            encoder.transform(&["orange"]),
            Err(AtlasMlError::InvalidArgument {
                op: "label_encoder_transform",
                reason: "label was not observed during fitting",
            })
        );
    }

    #[test]
    fn decodes_encoded_indices() {
        let encoder = LabelEncoder::fit(&[3, 1, 2]).unwrap();

        assert_eq!(encoder.inverse_transform(&[2, 0, 1]), Ok(vec![3, 1, 2]));
        assert_eq!(
            encoder.inverse_transform(&[3]),
            Err(AtlasMlError::InvalidArgument {
                op: "label_encoder_inverse_transform",
                reason: "encoded label index is out of bounds",
            })
        );
    }
}
