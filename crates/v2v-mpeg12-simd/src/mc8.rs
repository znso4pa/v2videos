unsafe extern "C" {
    fn v2v_mpeg12_mc8_neon(src: *const u8, step: usize, phase: u8, dst: *mut u8);
}

pub(crate) fn available() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}

pub(crate) fn predict(reference: &[u8], base: usize, step: usize, phase: u8) -> [u8; 64] {
    assert!(phase < 4);
    let width = 8 + usize::from(phase & 1);
    let last_row = 7 + usize::from(phase >> 1);
    let source = &reference[base..];
    assert!(last_row
        .checked_mul(step)
        .and_then(|offset| offset.checked_add(width))
        .is_some_and(|span| span <= source.len()));

    let mut dst = [0; 64];
    // SAFETY: install checks NEON before registering this private callback.
    // The phase and checked span cover all C reads, including the half-pel taps.
    // dst is a distinct initialized array, with exactly 64 writable bytes.
    unsafe { v2v_mpeg12_mc8_neon(source.as_ptr(), step, phase, dst.as_mut_ptr()) };
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_registers_c_prediction_when_neon_is_present() {
        crate::install();
        let expected = if available() { "c-neon" } else { "rust" };
        assert_eq!(v2v_mpeg12::decode::mc8_backend(), expected);
    }

    #[test]
    fn all_motion_phases_match_scalar_with_exact_input_spans() {
        if !available() {
            return;
        }
        let mut seed = 0xfa71_b52eu32;
        for step in [8, 9, 16, 23, 64, 127, 720, 1440] {
            for phase in 0..4u8 {
                let width = 8 + usize::from(phase & 1);
                let rows = 7 + usize::from(phase >> 1);
                for base in [0, 1, 15] {
                    let mut input = vec![0; base + rows * step + width];
                    for value in &mut input {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        *value = (seed >> 16) as u8;
                    }
                    let want = std::array::from_fn(|i| {
                        let at = base + i / 8 * step + i % 8;
                        let a = u16::from(input[at]);
                        match phase {
                            0 => a as u8,
                            1 => ((a + u16::from(input[at + 1]) + 1) >> 1) as u8,
                            2 => ((a + u16::from(input[at + step]) + 1) >> 1) as u8,
                            _ => {
                                ((a + u16::from(input[at + 1])
                                    + u16::from(input[at + step])
                                    + u16::from(input[at + step + 1])
                                    + 2)
                                    >> 2) as u8
                            }
                        }
                    });
                    assert_eq!(predict(&input, base, step, phase), want);
                }
            }
        }
    }
}
