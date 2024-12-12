use ndarray::{Array, Array2};

use crate::types::ArrayF32;

pub trait Activation {
    // to apply the activation
    fn apply(&self, arr: &ArrayF32) -> ArrayF32;
    // to apply the deriviative of the activation
    fn d_apply(&self, arr: &ArrayF32) -> ArrayF32;
}
pub struct ReLU;
pub struct Sigmoid;
pub struct Linear;

impl Activation for ReLU {
    fn apply(&self, arr: &ArrayF32) -> ArrayF32 {
        arr.mapv(|x| if x < 0 as f32 { 0 as f32 } else { x })
    }
    fn d_apply(&self, arr: &ArrayF32) -> ArrayF32 {
        arr.mapv(|x| if x < 0 as f32 { 0 as f32 } else { 1 as f32 })
    }
}
impl Activation for Sigmoid {
    fn apply(&self, arr: &ArrayF32) -> ArrayF32 {
        arr.mapv(|x| 1.0 / (1.0 + f32::exp(-x)))
    }

    fn d_apply(&self, arr: &ArrayF32) -> ArrayF32 {
        let sigmoid_arr = self.apply(arr);
        sigmoid_arr.clone() * (1 as f32 - sigmoid_arr)
    }
}

impl Activation for Linear {
    fn apply(&self, arr: &ArrayF32) -> ArrayF32 {
        arr.clone()
    }

    fn d_apply(&self, arr: &ArrayF32) -> ArrayF32 {
        arr.mapv(|_| 1.0)
    }
}
