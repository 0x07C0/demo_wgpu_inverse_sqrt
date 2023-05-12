#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::num_traits::Float;
use spirv_std::{glam::UVec3, spirv};

#[spirv(compute(threads(64)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] storage: &mut [f32],
) {
    let index = id.x as usize;
    if storage[index] == 0. {
        storage[index] = f32::NAN;
    } else {
        storage[index] = 1. / storage[index].sqrt();
    }
}
