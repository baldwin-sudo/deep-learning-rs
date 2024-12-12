mod activations;
mod criterions;
mod layers;
mod optimizers;

mod types;
use std::time::Instant;

use activations::{Activation, Linear, ReLU, Sigmoid};
use criterions::{Criterion, MSE};
use layers::{Dense, Network};
use ndarray::prelude::*;
use optimizers::{Optimizer, SGD};
fn main() {
    // XOR TEST
    let x = arr2(&[[1., 0.], [0., 1.], [1., 1.], [0., 0.]]);
    let y = arr2(&[[1.0], [1.0], [0.0], [0.0]]);
    // AND TEST
    // let x = arr2(&[[0., 0.], [0., 1.], [1., 0.], [1., 1.]]);
    // let y = arr2(&[[0.], [0.], [0.], [1.]]);
    // // OR TEST
    // let x = arr2(&[[0., 0.], [0., 1.], [1., 0.], [1., 1.]]);
    // let y = arr2(&[[0.], [1.], [1.], [1.]]);

    let x_shape = x.shape();
    println!("x shape {x_shape:?}");
    let criterion = MSE;

    let activation1 = ReLU;
    let activation2 = ReLU;
    let activation3 = ReLU;
    let mut layer1 = Dense::new((2, 3), Box::new(activation1));
    let mut layer2 = Dense::new((3, 3), Box::new(activation2));
    let mut layer3 = Dense::new((3, 3), Box::new(activation3));
    println!(" layer 1 :{layer1}");
    let activation4 = Sigmoid;
    let mut layer4 = Dense::new((3, 1), Box::new(activation4));
    println!(" layer 2 :{layer2}");

    let mut net = Network::new(vec![layer1, layer2, layer3, layer4]);

    let mut optimizer: SGD = SGD {
        learning_rate: 0.001,
    };
    let batch_size = 3;
    let epochs = 10000;
    // Start timing
    let start_time = Instant::now();
    net.train(
        x.clone(),
        y.clone(),
        &criterion,
        &mut optimizer,
        batch_size,
        epochs,
    ); // Stop timing
    let duration = start_time.elapsed();

    // Make predictions
    let predictions = net.forward(x.clone());

    // Threshold predictions
    let thresholded_predictions = predictions.mapv(|p| if p >= 0.5 { 1.0 } else { 0.0 });

    // Calculate accuracy
    let correct_predictions =
        (&thresholded_predictions - &y.clone()).mapv(|x| if x == 0.0 { 1.0 } else { 0.0 });
    let num_correct: usize = correct_predictions.sum() as usize;

    let accuracy = num_correct as f32 / y.shape()[0] as f32;
    println!("Accuracy: {:.2}%", accuracy * 100.0);
    println!("Training time: {:?}", duration);
}
