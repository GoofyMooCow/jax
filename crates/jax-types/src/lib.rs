//! Name resolution and type checking for Jax.

pub mod check;
pub mod ty;

pub use check::{check, CheckOutput};
pub use ty::{CallPlan, EnumInfo, ParamSource, Res, StructInfo, Target, Ty, TypeInfo, VariantInfo};
