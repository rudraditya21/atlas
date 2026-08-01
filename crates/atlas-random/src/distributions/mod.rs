mod normal;
mod uniform;

pub use normal::normal;
pub use uniform::uniform;

fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}
