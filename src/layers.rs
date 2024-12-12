use crate::activations::{self, Activation};
use crate::criterions::Criterion;
use crate::optimizers::Optimizer;
use crate::types::ArrayF32;
use ndarray::{array, Axis};
use rand::prelude::*;
pub struct Dense {
    pub activation: Box<dyn Activation>, // activation function
    pub x: Option<ArrayF32>,             // input that the layer got
    pub z: Option<ArrayF32>,             // the preactivation = weighted sum + bias

    pub weights: ArrayF32, // weights for all the neurons so its a (input_size,n_neurons)
    pub bias: ArrayF32,    // bias for all neurons (1,n_neurons,)
}
use std::f32::NEG_INFINITY;
use std::fmt;

impl fmt::Display for Dense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format each field of Dense
        write!(
            f,
            "Dense Layer:\n\
            Activation: {}\n\
            Weights:\n{}\n\
            Bias:\n{}\n\
            Last Input (x):\n{}\n\
            Last Preactivation (z):\n{}",
            // Attempt to display the activation function's name
            // Assuming activation implements Display, otherwise use placeholder
            "Custom Activation",
            self.weights,
            self.bias,
            match &self.x {
                Some(x) => format!("{}", x),
                None => "None".to_string(),
            },
            match &self.z {
                Some(z) => format!("{}", z),
                None => "None".to_string(),
            }
        )
    }
}

impl Dense {
    pub fn new((input_size, n_neurons): (usize, usize), activation: Box<dyn Activation>) -> Self {
        let mut rng = rand::thread_rng();

        let weights = ArrayF32::from_shape_fn((n_neurons, input_size), |_| rng.gen()); // replace with rng.gen()
        let bias = ArrayF32::from_shape_fn((1, n_neurons), |_| 0.0);
        Dense {
            activation,
            x: None,
            z: None,
            weights: weights,
            bias,
        }
    }
    pub fn forward(&mut self, input: ArrayF32) -> ArrayF32 {
        // input must be (batch_size,input_size)
        self.x = Some(input.clone());
        self.z = Some(self.x.clone().unwrap().dot(&self.weights.t()) + self.bias.clone());

        // output must be (batch_size,1);
        self.activation.apply(&self.z.clone().unwrap())
    }
    pub fn backward(&self, grad_next_layer: &ArrayF32) -> (ArrayF32, ArrayF32, ArrayF32) {
        // grad_next_layer corresponds to dL/dZ_next (gradient from the next layer)

        // Gradients of the preactivation (Z)
        let dZ = grad_next_layer * self.activation.d_apply(&self.z.clone().unwrap());

        // Gradient with respect to weights
        let dW = dZ.t().dot(&self.x.clone().unwrap()); // Align dimensions by transposing twice

        // Gradient with respect to bias
        let dB = dZ.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0)); // Sum along batch axis

        // Gradient with respect to inputs (for previous layer)
        let dX = dZ.dot(&self.weights);

        (dW, dB, dX)
    }
}
pub struct Network {
    pub layers: Vec<Dense>,
}
impl Network {
    pub fn new(layers: Vec<Dense>) -> Self {
        Network { layers }
    }
    pub fn forward(&mut self, x: ArrayF32) -> ArrayF32 {
        self.layers
            .iter_mut()
            .fold(x.clone(), |x, layer| layer.forward(x))
    }
    fn create_batches(data: &ArrayF32, batch_size: usize) -> Vec<ArrayF32> {
        data.axis_chunks_iter(Axis(0), batch_size)
            .map(|batch| batch.to_owned())
            .collect()
    }

    pub fn train(
        &mut self,
        x: ArrayF32,
        y: ArrayF32,
        criterion: &dyn Criterion,
        optimizer: &mut dyn Optimizer,
        batch_size: usize,
        epochs: usize,
    ) {
        for epoch in 0..epochs {
            // Create batches for x and y
            let x_batches = Self::create_batches(&x, batch_size);
            let y_batches = Self::create_batches(&y, batch_size);
            let mut error = array![[0.0]];
            for (x_batch, y_batch) in x_batches.iter().zip(y_batches.iter()) {
                let y_hat = self.forward(x_batch.clone());
                error = criterion.step(y_batch.clone(), y_hat.clone());
                let d_error = criterion.d_step(y_batch.clone(), y_hat);

                // Update the network using the optimizer
                optimizer.step(self, d_error);
                // Log error every 100 epochs
            }
            if epoch % 100 == 0 {
                println!("Epoch: {epoch}, Error: {error}");
            }
        }
    }
}
