#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    Default,
    Trust,
    ReadOnly,
    Yolo,
}
