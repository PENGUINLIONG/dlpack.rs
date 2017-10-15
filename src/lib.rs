#![allow(dead_code)]
///! For detailed documentation, see:
///! https://github.com/dmlc/dlpack
use std::os::raw::c_void;

pub type DeviceType = u32;
pub mod device_type {
    use super::DeviceType;

    const CPU:        DeviceType = 1;
    const GPU:        DeviceType = 2;
    const CPU_PINNED: DeviceType = 3;
    const OPENCL:     DeviceType = 4;
    const METAL:      DeviceType = 8;
    const VPI:        DeviceType = 9;
    const ROCM:       DeviceType = 10;
}

#[repr(C)]
pub struct Context {
    device_type: DeviceType,
    device_id: i32,
}

pub mod data_type_code {
    const INT: u8 = 0;
    const UINT: u8 = 1;
    const FLOAT: u8 = 2;
}

#[repr(C)]
pub struct DataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
pub struct Tensor {
    data: *mut c_void,
    ctx: Context,
    ndim: i32,
    dtype: DataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}
