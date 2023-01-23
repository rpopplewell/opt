
#![allow(unused)]

use argmin::core::{ArgminFloat, OptimizationResult};
use std::env;
extern crate argmin;
extern crate argmin_testfunctions;
use argmin_testfunctions::{
    rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian
};
use argmin::core::{Error, CostFunction, Gradient, Hessian};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::core::Executor;

fn main() {

    env::set_var("RUST_BACKTRACE", "1");

    /// First, we create a struct called `Rosenbrock` for your problem
    struct Rosenbrock {
        a: f64,
        b: f64,
    }

    /// Implement `CostFunction` for `Rosenbrock`
    ///
    /// First, we need to define the types which we will be using. Our parameter
    /// vector will be a `Vec` of `f64` values and our cost function value will 
    /// be a 64 bit floating point value.
    /// This is reflected in the associated types `Param` and `Output`, respectively.
    ///
    /// The method `cost` then defines how the cost function is computed for a
    /// parameter vector `p`. Note that we have access to the fields `a` and `b`
    /// of `Rosenbrock`.
    impl CostFunction for Rosenbrock {
        /// Type of the parameter vector
        type Param = Vec<f64>;
        /// Type of the return value computed by the cost function
        type Output = f64;

        /// Apply the cost function to a parameter `p`
        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            // Evaluate 2D Rosenbrock function
            Ok(rosenbrock_2d(p, self.a, self.b))
        }
    }

    /// Implement `Gradient` for `Rosenbrock`
    ///
    /// Similarly to `CostFunction`, we need to define the type of our parameter
    /// vectors and of the gradient we are computing. Since the gradient is also
    /// a vector, it is of type `Vec<f64>` just like `Param`.
    impl Gradient for Rosenbrock {
        /// Type of the parameter vector
        type Param = Vec<f64>;
        /// Type of the gradient
        type Gradient = Vec<f64>;

        /// Compute the gradient at parameter `p`.
        fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
            // Compute gradient of 2D Rosenbrock function
            Ok(rosenbrock_2d_derivative(p, self.a, self.b))
        }
    }

    /// Implement `Hessian` for `Rosenbrock`
    ///
    /// Again the types of the involved parameter vector and the Hessian needs to
    /// be defined. Since the Hessian is a 2D matrix, we use `Vec<Vec<f64>>` here.
    impl Hessian for Rosenbrock {
        /// Type of the parameter vector
        type Param = Vec<f64>;
        /// Type of the Hessian
        type Hessian = Vec<Vec<f64>>;

        /// Compute the Hessian at parameter `p`.
        fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
            // Compute Hessian of 2D Rosenbrock function
            let t = rosenbrock_2d_hessian(p, self.a, self.b);
            // Reshape the output
            Ok(vec![vec![t[0], t[1]], vec![t[2], t[3]]])
        }
    }

    let init_param = vec![1.0, -2.0];

    let cost = Rosenbrock { a: (1.0), b: (100.0) };
    let linesearch: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> = MoreThuenteLineSearch::new();
    let solver = SteepestDescent::new(linesearch);

    let res = Executor::new(cost, solver)
        // Via `configure`, one has access to the internally used state.
        // This state can be initialized, for instance by providing an
        // initial parameter vector.
        // The maximum number of iterations is also set via this method.
        // In this particular case, the state exposed is of type `IterState`.
        // The documentation of `IterState` shows how this struct can be
        // manipulated.
        // Population based solvers use `PopulationState` instead of 
        // `IterState`.
        .configure(|state|
            state
                // Set initial parameters (depending on the solver,
                // this may be required)
                .param(init_param)
                // Set maximum iterations to 10
                // (optional, set to `std::u64::MAX` if not provided)
                .max_iters(1000)
                // Set target cost. The solver stops when this cost
                // function value is reached (optional)
                .target_cost(0.0)
        )
        // run the solver on the defined problem
        .run();

    let res = match res {
        Ok(res) => res,
        Err(err) => !panic!("{}", err),
    };

    // print result
    println!("{}", res);

}
