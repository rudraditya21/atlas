//! Shared runtime-checked casts for Atlas SIMD implementations.

use std::{
    any::TypeId,
    mem::{align_of, size_of},
};

#[inline]
pub fn is_f32<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

#[inline]
pub fn is_f64<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

#[inline]
pub fn cast_value<U, T>(value: U) -> T
where
    U: Copy + 'static,
    T: Copy + 'static,
{
    assert_exact_type::<U, T>();
    // SAFETY: The runtime type, size, and alignment checks guarantee identical layouts.
    unsafe { std::mem::transmute_copy::<U, T>(&value) }
}

#[inline]
pub fn cast_slice<T: 'static, U: 'static>(data: &[T]) -> &[U] {
    assert_exact_type::<T, U>();
    // SAFETY: The runtime type, size, and alignment checks guarantee identical layouts.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<U>(), data.len()) }
}

#[inline]
pub fn cast_mut_slice<T: 'static, U: 'static>(data: &mut [T]) -> &mut [U] {
    assert_exact_type::<T, U>();
    // SAFETY: The runtime type, size, and alignment checks guarantee identical layouts.
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<U>(), data.len()) }
}

#[inline]
fn assert_exact_type<T: 'static, U: 'static>() {
    assert_eq!(TypeId::of::<T>(), TypeId::of::<U>());
    assert_eq!(size_of::<T>(), size_of::<U>());
    assert_eq!(align_of::<T>(), align_of::<U>());
}
