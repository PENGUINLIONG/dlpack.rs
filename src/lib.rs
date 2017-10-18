#![allow(dead_code)]
///! For detailed documentation, see:
///! https://github.com/dmlc/dlpack
use std::os::raw::c_void;

pub type DeviceTypeCode = i32;
pub mod device_type_codes {
    use super::DeviceTypeCode;

    pub const CPU:        DeviceTypeCode = 1;
    pub const GPU:        DeviceTypeCode = 2;
    pub const CPU_PINNED: DeviceTypeCode = 3;
    pub const OPENCL:     DeviceTypeCode = 4;
    pub const METAL:      DeviceTypeCode = 8;
    pub const VPI:        DeviceTypeCode = 9;
    pub const ROCM:       DeviceTypeCode = 10;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Context {
    pub device_type: DeviceTypeCode,
    pub device_id: i32,
}

pub type DataTypeCode = u8;
pub mod data_type_codes {
    use super::DataTypeCode;

    pub const INT:   DataTypeCode = 0;
    pub const UINT:  DataTypeCode = 1;
    pub const FLOAT: DataTypeCode = 2;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DataType {
    pub code: DataTypeCode,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
pub struct Tensor {
    pub data: *mut c_void,
    pub ctx: Context,
    pub ndim: i32,
    pub dtype: DataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}
