//! 只给测试用的 MSB 优先位写入器 —— 用来手工拼一个位流去喂解析器。
//!
//! 实现就在 `crate::bitwriter`（编码器也用同一个），这里只是给测试一个短名字。

pub use crate::bitwriter::BitWriter;
