use crate::types::ArrayF32;

pub trait Criterion {
    // calculate the error
    fn step(&self, y_true: ArrayF32, y_hat: ArrayF32) -> ArrayF32;
    // calculate the gradient of error with respect to prediction yhat
    fn d_step(&self, y_true: ArrayF32, y_hat: ArrayF32) -> ArrayF32;
}
pub struct MSE;

impl Criterion for MSE {
    fn step(&self, y_true: ArrayF32, y_hat: ArrayF32) -> ArrayF32 {
        assert_eq!(
            y_true.shape(),
            y_hat.shape(),
            "Shapes of y_true and y_hat must match!"
        );

        // Compute the mean squared error
        let mse = (y_true - y_hat)
            .mapv(|x| x.powi(2))
            .mean_axis(ndarray::Axis(0))
            .unwrap();

        // Convert to 2D array with shape (batch_size, 1) for consistency
        mse.insert_axis(ndarray::Axis(1))
    }

    fn d_step(&self, y_true: ArrayF32, y_hat: ArrayF32) -> ArrayF32 {
        assert_eq!(
            y_true.shape(),
            y_hat.shape(),
            "Shapes of y_true and y_hat must match!"
        );

        // Compute the mean squared error
        let d_mse = (y_true - y_hat)
            .mapv(|x| -2.0 * x)
            .mean_axis(ndarray::Axis(1))
            .unwrap();

        // Convert to 2D array with shape (batch_size, 1) for consistency
        d_mse.insert_axis(ndarray::Axis(1))
    }
}
