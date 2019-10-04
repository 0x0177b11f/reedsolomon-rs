// Implementation of GF(2^8): the finite field with 2^8 elements.
include!(concat!(env!("OUT_DIR"), "/table.rs"));

use crate::field::Field;

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct GF8Field;

impl Field for GF8Field {
    const ORDER: usize = 256;
    type Elem = u8;

    fn add(a: u8, b: u8) -> u8 {
        gal_add(a, b)
    }

    fn mul(a: u8, b: u8) -> u8 {
        gal_mul(a, b)
    }

    fn div(a: u8, b: u8) -> u8 {
        gal_div(a, b)
    }

    fn exp(elem: u8, n: usize) -> u8 {
        gal_exp(elem, n)
    }

    fn zero() -> u8 {
        0
    }

    fn one() -> u8 {
        1
    }

    fn nth_internal(n: usize) -> u8 {
        n as u8
    }

    fn mul_slice(c: u8, input: &[u8], out: &mut [u8]) {
        gal_mul_slice(c, input, out)
    }

    fn mul_slice_add(c: u8, input: &[u8], out: &mut [u8]) {
        gal_mul_slice_xor(c, input, out)
    }
}

macro_rules! return_if_empty {
    ($len:expr) => {
        if $len == 0 {
            return;
        }
    };
}

fn gal_add(a: u8, b: u8) -> u8 {
    a ^ b
}

fn gal_mul(a: u8, b: u8) -> u8 {
    MUL_TABLE[a as usize][b as usize]
}


fn gal_div(a: u8, b: u8) -> u8 {
    if a == 0 {
        return 0;
    }
    
    if b == 0 {
        panic!("Divisor is 0")
    }

    let log_a = LOG_TABLE[a as usize] as i32;
    let log_b = LOG_TABLE[b as usize] as i32;
    let mut log_result = log_a - log_b;

    if log_result < 0 {
        log_result += 255
    }
    return EXP_TABLE[log_result as usize];
}

fn gal_exp(a: u8, n: usize) -> u8 {
    if n == 0 {
        1
    } else if a == 0 {
        0
    } else {
        let log_a = LOG_TABLE[a as usize];
        let mut log_result = log_a as usize * n;
        while 255 <= log_result {
            log_result -= 255;
        }
        EXP_TABLE[log_result]
    }
}

fn gal_mul_slice(c: u8, input: &[u8], out: &mut [u8]) {
    assert_eq!(input.len(), out.len());

    let len = input.len();
    return_if_empty!(len);

    let mt = &MUL_TABLE[c as usize];

    for n in 0..len {
        out[n] = mt[input[n] as usize]
    }
}

fn gal_mul_slice_xor(c: u8, input: &[u8], out: &mut [u8]) {
    assert_eq!(input.len(), out.len());

    let len = input.len();
    return_if_empty!(len);

    let mt = &MUL_TABLE[c as usize];

    for n in 0..len {
        out[n] ^= mt[input[n] as usize];
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_associativity() {
        for i in 0..255 {
            let a = i as u8;

            for j in 0..255 {
                let b = j as u8;

                for k in 0..255 {
                    let c = k as u8;

                    let mut x = gal_add(a, gal_add(b, c));
                    let mut y = gal_add(gal_add(a, b), c);

                    if x != y {
                        println!("add does not match: {} != {}", x, y);
                        assert_eq!(x, y);
                    }

                    x = gal_mul(a, gal_mul(b, c));
                    y = gal_mul(gal_mul(a, b), c);
                    if x != y {
                        println!("multiply does not match: {} != {}", x, y);
                        assert_eq!(x, y);
                    }
                }
            }
        }
    }

    #[test]
    fn test_identity() {
        for i in 0..255 {
            let a = i as u8;
            let mut b = gal_add(a, 0);
            if a != b {
                println!("Add zero should yield same result: {} != {} ", a, b);
                assert_eq!(a, b);
            }

            b = gal_mul(a, 1);
            if a != b {
                println!("Mul by one should yield same result {} != {}", a, b);
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn test_commutativity() {
        for a in 0..256 {
            let a = a as u8;
            for b in 0..256 {
                let b = b as u8;
                let x = gal_add(a, b);
                let y = gal_add(b, a);
                assert_eq!(x, y);
                let x = gal_mul(a, b);
                let y = gal_mul(b, a);
                assert_eq!(x, y);
            }
        }
    }

    #[test]
    fn test_distributivity() {
        for a in 0..256 {
            let a = a as u8;
            for b in 0..256 {
                let b = b as u8;
                for c in 0..256 {
                    let c = c as u8;
                    let x = gal_mul(a, gal_add(b, c));
                    let y = gal_add(gal_mul(a, b), gal_mul(a, c));
                    assert_eq!(x, y);
                }
            }
        }
    }

    #[test]
    fn test_exp() {
        for a in 0..256 {
            let a = a as u8;
            let mut power = 1u8;
            for j in 0..256 {
                let x = gal_exp(a, j);
                assert_eq!(x, power);
                power = gal_mul(power, a);
            }
        }
    }

    #[test]
    fn test_galois() {
        assert_eq!(gal_mul(3, 4), 12);
        assert_eq!(gal_mul(7, 7), 21);
        assert_eq!(gal_mul(23, 45), 41);

        let input = [
            0, 1, 2, 3, 4, 5, 6, 10, 50, 100, 150, 174, 201, 255, 99, 32, 67, 85, 200, 199, 198,
            197, 196, 195, 194, 193, 192, 191, 190, 189, 188, 187, 186, 185,
        ];
        let mut output1 = vec![0; input.len()];
        let mut output2 = vec![0; input.len()];
        gal_mul_slice(25, &input, &mut output1);
        let expect = [
            0x0, 0x19, 0x32, 0x2b, 0x64, 0x7d, 0x56, 0xfa, 0xb8, 0x6d, 0xc7, 0x85, 0xc3, 0x1f,
            0x22, 0x7, 0x25, 0xfe, 0xda, 0x5d, 0x44, 0x6f, 0x76, 0x39, 0x20, 0xb, 0x12, 0x11, 0x8,
            0x23, 0x3a, 0x75, 0x6c, 0x47,
        ];
        for i in 0..input.len() {
            assert_eq!(expect[i], output1[i]);
        }
        gal_mul_slice(25, &input, &mut output2);
        for i in 0..input.len() {
            assert_eq!(expect[i], output2[i]);
        }

        let expect_xor = [
            0x0, 0x2d, 0x5a, 0x77, 0xb4, 0x99, 0xee, 0x2f, 0x79, 0xf2, 0x7, 0x51, 0xd4, 0x19, 0x31,
            0xc9, 0xf8, 0xfc, 0xf9, 0x4f, 0x62, 0x15, 0x38, 0xfb, 0xd6, 0xa1, 0x8c, 0x96, 0xbb,
            0xcc, 0xe1, 0x22, 0xf, 0x78,
        ];
        gal_mul_slice_xor(52, &input, &mut output1);
        for i in 0..input.len() {
            assert_eq!(expect_xor[i], output1[i]);
        }
        gal_mul_slice_xor(52, &input, &mut output2);
        for i in 0..input.len() {
            assert_eq!(expect_xor[i], output2[i]);
        }

        let expect = [
            0x0, 0xb1, 0x7f, 0xce, 0xfe, 0x4f, 0x81, 0x9e, 0x3, 0x6, 0xe8, 0x75, 0xbd, 0x40, 0x36,
            0xa3, 0x95, 0xcb, 0xc, 0xdd, 0x6c, 0xa2, 0x13, 0x23, 0x92, 0x5c, 0xed, 0x1b, 0xaa,
            0x64, 0xd5, 0xe5, 0x54, 0x9a,
        ];
        gal_mul_slice(177, &input, &mut output1);
        for i in 0..input.len() {
            assert_eq!(expect[i], output1[i]);
        }
        gal_mul_slice(177, &input, &mut output2);
        for i in 0..input.len() {
            assert_eq!(expect[i], output2[i]);
        }

        let expect_xor = [
            0x0, 0xc4, 0x95, 0x51, 0x37, 0xf3, 0xa2, 0xfb, 0xec, 0xc5, 0xd0, 0xc7, 0x53, 0x88,
            0xa3, 0xa5, 0x6, 0x78, 0x97, 0x9f, 0x5b, 0xa, 0xce, 0xa8, 0x6c, 0x3d, 0xf9, 0xdf, 0x1b,
            0x4a, 0x8e, 0xe8, 0x2c, 0x7d,
        ];
        gal_mul_slice_xor(117, &input, &mut output1);
        for i in 0..input.len() {
            assert_eq!(expect_xor[i], output1[i]);
        }
        gal_mul_slice_xor(117, &input, &mut output2);
        for i in 0..input.len() {
            assert_eq!(expect_xor[i], output2[i]);
        }

        assert_eq!(gal_exp(2, 2), 4);
        assert_eq!(gal_exp(5, 20), 235);
        assert_eq!(gal_exp(13, 7), 43);
    }

    #[test]
    fn test_div_a_is_0() {
        assert_eq!(0, gal_div(0, 100));
    }

    #[test]
    #[should_panic]
    fn test_div_b_is_0() {
        gal_div(1, 0);
    }
}
