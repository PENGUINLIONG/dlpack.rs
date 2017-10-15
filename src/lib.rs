#![allow(dead_code)]
///! For detailed documentation, see:
///! https://github.com/dmlc/dlpack
use std::os::raw::c_void;

pub type DeviceType = u32;
pub mod device_type {
    use super::DeviceType;

    pub const CPU:        DeviceType = 1;
    pub const GPU:        DeviceType = 2;
    pub const CPU_PINNED: DeviceType = 3;
    pub const OPENCL:     DeviceType = 4;
    pub const METAL:      DeviceType = 8;
    pub const VPI:        DeviceType = 9;
    pub const ROCM:       DeviceType = 10;
}

#[repr(C)]
pub struct Context {
    device_type: DeviceType,
    device_id: i32,
}

pub mod data_type_code {
    pub const INT: u8 = 0;
    pub const UINT: u8 = 1;
    pub const FLOAT: u8 = 2;
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
