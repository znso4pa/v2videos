use std::ffi::{c_char, c_double, c_int, c_long, c_void, CStr};
use std::ptr::{null, NonNull};
use std::sync::OnceLock;

use v2v_mp3::{Error, Framer, Header, Pcm, Result};

// libmpg123 公共头文件 mpg123.h / fmt123.h。调用边界集中在本模块。
extern "C" {
    fn mpg123_init() -> c_int;
    fn mpg123_new(decoder: *const c_char, error: *mut c_int) -> *mut c_void;
    fn mpg123_delete(handle: *mut c_void);
    fn mpg123_param2(handle: *mut c_void, key: c_int, value: c_long, fvalue: c_double) -> c_int;
    fn mpg123_format_none(handle: *mut c_void) -> c_int;
    fn mpg123_format(handle: *mut c_void, rate: c_long, channels: c_int, encoding: c_int) -> c_int;
    fn mpg123_getformat(
        handle: *mut c_void,
        rate: *mut c_long,
        channels: *mut c_int,
        encoding: *mut c_int,
    ) -> c_int;
    fn mpg123_open_feed(handle: *mut c_void) -> c_int;
    fn mpg123_decode(
        handle: *mut c_void,
        input: *const u8,
        input_len: usize,
        output: *mut u8,
        output_len: usize,
        written: *mut usize,
    ) -> c_int;
    fn mpg123_plain_strerror(code: c_int) -> *const c_char;
}

const OK: c_int = 0;
const ERROR: c_int = -1;
const NEED_MORE: c_int = -10;
const NEW_FORMAT: c_int = -11;
const FLAGS: c_int = 1;
const QUIET: c_long = 0x20;
const NO_RESYNC: c_long = 0x80;
const IGNORE_INFOFRAME: c_long = 0x4000;
const NO_READAHEAD: c_long = 0x400000;
const FLOAT32: c_int = 0x200;

/// libmpg123 的 Layer III 解码器，使用与纯 Rust 后端相同的分帧和 PCM 刻度。
pub struct Mp3Decoder {
    framer: Framer,
    handle: Handle,
}

impl Mp3Decoder {
    pub fn new(channels: usize, sample_rate: u32) -> Result<Self> {
        let framer = Framer::new(channels, sample_rate)?;
        let handle = Handle::new(channels, sample_rate)?;
        Ok(Self { framer, handle })
    }

    pub fn feed(&mut self, data: &[u8]) -> Result<Vec<Pcm>> {
        let mut out = Vec::new();
        self.framer.feed(data, |header, frame| {
            out.push(self.handle.decode(header, frame)?);
            Ok(())
        })?;
        Ok(out)
    }

    pub fn finish(&self) -> Result<()> {
        self.framer.finish()
    }
}

struct Handle(NonNull<c_void>);

// 每个实例独占一个 C 句柄，调用需要 &mut。跨线程移动不会与其他线程共享句柄。
unsafe impl Send for Handle {}

impl Handle {
    fn new(channels: usize, sample_rate: u32) -> Result<Self> {
        static INIT: OnceLock<c_int> = OnceLock::new();
        // 初始化是进程级资源。保留到进程结束，避免与独立句柄的调用重叠。
        check(*INIT.get_or_init(|| unsafe { mpg123_init() }))?;
        let mut err = ERROR;
        let raw = unsafe { mpg123_new(null(), &mut err) };
        let handle = Self(NonNull::new(raw).ok_or_else(|| error(err))?);
        let raw = handle.0.as_ptr();
        // 共享分帧器已验证帧边界。关闭库自己的自动同步、前看与文件层 gapless。
        unsafe {
            check(mpg123_param2(
                raw,
                FLAGS,
                QUIET | NO_RESYNC | IGNORE_INFOFRAME | NO_READAHEAD,
                0.0,
            ))?;
            check(mpg123_format_none(raw))?;
            check(mpg123_format(
                raw,
                sample_rate as c_long,
                channels as c_int,
                FLOAT32,
            ))?;
            check(mpg123_open_feed(raw))?;
        }
        Ok(handle)
    }

    fn decode(&mut self, h: Header, frame: &[u8]) -> Result<Pcm> {
        let raw = self.0.as_ptr();
        let mut pcm = [0.0_f32; 2304];
        let expected = h.samples * h.channels * size_of::<f32>();
        let mut completed = 0;
        let mut input = frame;
        // 首帧可能先宣布格式，然后才交出数据。
        for _ in 0..3 {
            let mut written = 0;
            let code = unsafe {
                mpg123_decode(
                    raw,
                    input.as_ptr(),
                    input.len(),
                    pcm.as_mut_ptr().cast::<u8>().add(completed),
                    expected - completed,
                    &mut written,
                )
            };
            input = &[];
            if written > expected - completed {
                return Err(Error::Bad("mpg123 PCM length"));
            }
            completed += written;
            if code == NEW_FORMAT {
                let (mut rate, mut channels, mut encoding) = (0, 0, 0);
                unsafe {
                    check(mpg123_getformat(
                        raw,
                        &mut rate,
                        &mut channels,
                        &mut encoding,
                    ))?
                };
                if rate != h.sample_rate as c_long
                    || channels as usize != h.channels
                    || encoding != FLOAT32
                {
                    return Err(Error::Bad("mpg123 PCM format"));
                }
            } else if !matches!(code, OK | NEED_MORE) {
                return Err(error(code));
            }
            if completed == expected {
                let mut planes: Pcm = (0..h.channels)
                    .map(|_| Vec::with_capacity(h.samples))
                    .collect();
                for point in pcm[..h.samples * h.channels].chunks_exact(h.channels) {
                    for (plane, sample) in planes.iter_mut().zip(point) {
                        plane.push(sample * 32768.0);
                    }
                }
                return Ok(planes);
            }
            if code == NEED_MORE {
                break;
            }
        }
        Err(Error::Bad("mpg123 returned an incomplete PCM frame"))
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { mpg123_delete(self.0.as_ptr()) };
    }
}

fn check(code: c_int) -> Result<()> {
    if code == OK {
        Ok(())
    } else {
        Err(error(code))
    }
}

fn error(code: c_int) -> Error {
    let message = unsafe { CStr::from_ptr(mpg123_plain_strerror(code)) }.to_string_lossy();
    Error::Unsupported(format!("mpg123 ({code}): {message}"))
}
