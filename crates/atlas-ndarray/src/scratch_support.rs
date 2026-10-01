use std::{cell::RefCell, thread::LocalKey};

#[doc(hidden)]
pub fn with_thread_local_buffer<T, R>(
    storage: &'static LocalKey<RefCell<Vec<T>>>,
    len: usize,
    fill: T,
    operation: impl FnOnce(&mut Vec<T>) -> R,
) -> R
where
    T: Clone + 'static,
{
    storage.with(|storage| {
        let mut buffer = storage.take();
        buffer.clear();
        buffer.resize(len, fill);
        let result = operation(&mut buffer);
        buffer.clear();
        storage.replace(buffer);
        result
    })
}
