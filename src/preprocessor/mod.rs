mod base;
mod lower_case;
mod start_end;

pub use base::PreProcessor;
pub use lower_case::LowerCasePreProcessor;
pub use start_end::StartEndTokensPreProcessor;

pub const START_OF_STRING: &str = "<s>";
pub const END_OF_STRING: &str = "<s/>";
