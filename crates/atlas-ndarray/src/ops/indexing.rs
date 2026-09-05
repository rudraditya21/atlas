use crate::{ArrayElement, ArrayView, NDArray};

mod sealed {
    pub trait Sealed {}

    macro_rules! impl_sealed {
        ($($ty:ty),+ $(,)?) => { $(impl Sealed for $ty {})+ };
    }

    impl_sealed!(bool, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);
}

/// Defines whether an ndarray element is selected by [`NDArray::nonzero`] and [`NDArray::argwhere`].
pub trait Truthy: ArrayElement + sealed::Sealed {
    /// Returns whether this value is nonzero or truthy.
    fn is_truthy(self) -> bool;
}

macro_rules! impl_truthy {
    ($($ty:ty),+ $(,)?) => { $(
        impl Truthy for $ty {
            fn is_truthy(self) -> bool { self != 0 as $ty }
        }
    )+ };
}

impl_truthy!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);

impl Truthy for bool {
    fn is_truthy(self) -> bool {
        self
    }
}

fn coordinates<'a, T: Truthy + 'a>(
    shape: &[usize],
    values: impl Iterator<Item = &'a T>,
) -> NDArray<usize> {
    let mut coordinates = Vec::new();
    let mut count = 0;
    for (linear, value) in values.enumerate() {
        if value.is_truthy() {
            let mut index = linear;
            for &dimension in shape.iter().rev() {
                coordinates.push(index % dimension);
                index /= dimension;
            }
            let start = coordinates.len() - shape.len();
            coordinates[start..].reverse();
            count += 1;
        }
    }
    NDArray::from_row_major_parts(vec![count, shape.len()], coordinates)
        .expect("nonzero coordinates preserve ndarray invariants")
}

macro_rules! impl_indexing {
    ($operand:ty) => {
        impl<T: Truthy> $operand {
            /// Returns row-major coordinates for all nonzero or truthy elements.
            pub fn nonzero(&self) -> NDArray<usize> {
                coordinates(self.shape(), self.iter())
            }

            /// Alias for [`NDArray::nonzero`].
            pub fn argwhere(&self) -> NDArray<usize> {
                self.nonzero()
            }
        }
    };
}

impl_indexing!(NDArray<T>);
impl_indexing!(ArrayView<'_, T>);

#[cfg(test)]
mod tests {
    use crate::NDArray;

    #[test]
    fn nonzero_returns_row_major_coordinates() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 2, 0, 4, 0, 6]).unwrap();

        assert_eq!(array.nonzero().shape(), &[3, 2]);
        assert_eq!(array.nonzero().data(), &[0, 1, 1, 0, 1, 2]);
        assert_eq!(array.argwhere().data(), array.nonzero().data());
    }

    #[test]
    fn nonzero_uses_logical_view_order_and_boolean_truthiness() {
        let array =
            NDArray::from_shape_vec([2, 3], vec![false, true, false, true, false, false]).unwrap();
        let transposed = array.view().transpose();

        assert_eq!(transposed.nonzero().shape(), &[2, 2]);
        assert_eq!(transposed.nonzero().data(), &[0, 1, 1, 0]);
    }
}
