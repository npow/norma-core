pub mod protocol;
pub mod presets;

pub mod st3215_proto {
    include!("proto/st3215.rs");
}

mod driver;
mod port;
mod port_meta;

mod calibrate;
mod calibration_store;
mod state;
mod auto_calibrate;

#[cfg(test)]
mod calibrate_test;
mod errors;

pub use driver::{start_st3215_driver, St3215Driver};
