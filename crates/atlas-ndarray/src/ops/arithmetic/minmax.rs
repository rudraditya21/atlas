use crate::Numeric;

mod sealed {
    pub trait Sealed {}

    macro_rules! impl_sealed {
        ($($ty:ty),+ $(,)?) => { $(impl Sealed for $ty {})+ };
    }

    impl_sealed!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);
}

pub trait ElementwiseMinMax: Numeric + sealed::Sealed {
    fn elementwise_min(self, other: Self) -> Self;
    fn elementwise_max(self, other: Self) -> Self;
}

macro_rules! impl_ordered_minmax {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ElementwiseMinMax for $ty {
                fn elementwise_min(self, other: Self) -> Self { self.min(other) }
                fn elementwise_max(self, other: Self) -> Self { self.max(other) }
            }
        )+
    };
}

macro_rules! impl_float_minmax {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ElementwiseMinMax for $ty {
                fn elementwise_min(self, other: Self) -> Self { self.min(other) }
                fn elementwise_max(self, other: Self) -> Self { self.max(other) }
            }
        )+
    };
}

impl_ordered_minmax!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
impl_float_minmax!(f32, f64);
