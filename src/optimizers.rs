use crate::activations;
use crate::layers::Network;
use crate::types::ArrayF32;

pub trait Optimizer {
    fn step(&mut self, network: &mut Network, grad_error: ArrayF32);
}
pub struct SGD {
    pub learning_rate: f32,
}
impl Optimizer for SGD {
    fn step(&mut self, network: &mut Network, grad_error: ArrayF32) {
        let mut dX = grad_error;
        for layer in network.layers.iter_mut().rev() {
            let (dW, dB, dX) = layer.backward(&dX);
            let shape_dw = dW.shape();
            let shape_db = dB.shape();
            // println!("Weights shape: {:?}", layer.weights.shape());
            // println!("Bias shape: {:?}", layer.bias.shape());
            // println!(
            //     "Gradient shapes - dW: {:?}, dB: {:?}",
            //     dW.shape(),
            //     dB.shape()
            // );

            layer.weights = layer.weights.clone() - self.learning_rate * dW;
            layer.bias = layer.bias.clone() - self.learning_rate * dB;
        }
    }
}
