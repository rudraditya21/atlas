pub(crate) mod broadcast;
pub(crate) mod reshape;
pub(crate) mod stride;
pub(crate) mod transpose;

pub use broadcast::{
    BroadcastMetadata, broadcast_pair, broadcast_shape, broadcast_strides,
    contiguous_broadcast_metadata,
};
pub use stride::{compute_strides, element_count};
