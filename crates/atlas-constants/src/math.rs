pub mod f32 {
    pub const E: f32 = core::f32::consts::E;
    pub const EULER_GAMMA: f32 = 0.57721566490153286060651209008240243104215933593992_f32;
    pub const FRAC_1_PI: f32 = core::f32::consts::FRAC_1_PI;
    pub const FRAC_1_SQRT_2: f32 = core::f32::consts::FRAC_1_SQRT_2;
    pub const FRAC_2_PI: f32 = core::f32::consts::FRAC_2_PI;
    pub const FRAC_2_SQRT_PI: f32 = core::f32::consts::FRAC_2_SQRT_PI;
    pub const FRAC_PI_2: f32 = core::f32::consts::FRAC_PI_2;
    pub const FRAC_PI_3: f32 = core::f32::consts::FRAC_PI_3;
    pub const FRAC_PI_4: f32 = core::f32::consts::FRAC_PI_4;
    pub const FRAC_PI_6: f32 = core::f32::consts::FRAC_PI_6;
    pub const FRAC_PI_8: f32 = core::f32::consts::FRAC_PI_8;
    pub const INFINITY: f32 = f32::INFINITY;
    pub const LN_2: f32 = core::f32::consts::LN_2;
    pub const LN_10: f32 = core::f32::consts::LN_10;
    pub const LOG2_E: f32 = core::f32::consts::LOG2_E;
    pub const LOG10_E: f32 = core::f32::consts::LOG10_E;
    pub const NEG_INFINITY: f32 = f32::NEG_INFINITY;
    pub const NAN: f32 = f32::NAN;
    pub const PI: f32 = core::f32::consts::PI;
    pub const SQRT_2: f32 = core::f32::consts::SQRT_2;
    pub const TAU: f32 = core::f32::consts::TAU;
}

pub mod f64 {
    pub const E: f64 = core::f64::consts::E;
    pub const EULER_GAMMA: f64 = 0.57721566490153286060651209008240243104215933593992_f64;
    pub const FRAC_1_PI: f64 = core::f64::consts::FRAC_1_PI;
    pub const FRAC_1_SQRT_2: f64 = core::f64::consts::FRAC_1_SQRT_2;
    pub const FRAC_2_PI: f64 = core::f64::consts::FRAC_2_PI;
    pub const FRAC_2_SQRT_PI: f64 = core::f64::consts::FRAC_2_SQRT_PI;
    pub const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
    pub const FRAC_PI_3: f64 = core::f64::consts::FRAC_PI_3;
    pub const FRAC_PI_4: f64 = core::f64::consts::FRAC_PI_4;
    pub const FRAC_PI_6: f64 = core::f64::consts::FRAC_PI_6;
    pub const FRAC_PI_8: f64 = core::f64::consts::FRAC_PI_8;
    pub const INFINITY: f64 = f64::INFINITY;
    pub const LN_2: f64 = core::f64::consts::LN_2;
    pub const LN_10: f64 = core::f64::consts::LN_10;
    pub const LOG2_E: f64 = core::f64::consts::LOG2_E;
    pub const LOG10_E: f64 = core::f64::consts::LOG10_E;
    pub const NEG_INFINITY: f64 = f64::NEG_INFINITY;
    pub const NAN: f64 = f64::NAN;
    pub const PI: f64 = core::f64::consts::PI;
    pub const SQRT_2: f64 = core::f64::consts::SQRT_2;
    pub const TAU: f64 = core::f64::consts::TAU;
}

#[cfg(test)]
mod tests {
    #[test]
    fn f32_constants_match_core() {
        assert_eq!(super::f32::TAU, core::f32::consts::TAU);
        assert_eq!(super::f32::LOG2_E, core::f32::consts::LOG2_E);
        assert_eq!(super::f32::LOG10_E, core::f32::consts::LOG10_E);
        assert_eq!(super::f32::LN_2, core::f32::consts::LN_2);
        assert_eq!(super::f32::LN_10, core::f32::consts::LN_10);
        assert_eq!(super::f32::SQRT_2, core::f32::consts::SQRT_2);
        assert_eq!(super::f32::FRAC_1_SQRT_2, core::f32::consts::FRAC_1_SQRT_2);
        assert_eq!(super::f32::FRAC_2_SQRT_PI, core::f32::consts::FRAC_2_SQRT_PI);
        assert_eq!(super::f32::FRAC_1_PI, core::f32::consts::FRAC_1_PI);
        assert_eq!(super::f32::FRAC_2_PI, core::f32::consts::FRAC_2_PI);
        assert_eq!(super::f32::FRAC_PI_2, core::f32::consts::FRAC_PI_2);
        assert_eq!(super::f32::FRAC_PI_3, core::f32::consts::FRAC_PI_3);
        assert_eq!(super::f32::FRAC_PI_4, core::f32::consts::FRAC_PI_4);
        assert_eq!(super::f32::FRAC_PI_6, core::f32::consts::FRAC_PI_6);
        assert_eq!(super::f32::FRAC_PI_8, core::f32::consts::FRAC_PI_8);
    }

    #[test]
    fn f64_constants_match_core() {
        assert_eq!(super::f64::TAU, core::f64::consts::TAU);
        assert_eq!(super::f64::LOG2_E, core::f64::consts::LOG2_E);
        assert_eq!(super::f64::LOG10_E, core::f64::consts::LOG10_E);
        assert_eq!(super::f64::LN_2, core::f64::consts::LN_2);
        assert_eq!(super::f64::LN_10, core::f64::consts::LN_10);
        assert_eq!(super::f64::SQRT_2, core::f64::consts::SQRT_2);
        assert_eq!(super::f64::FRAC_1_SQRT_2, core::f64::consts::FRAC_1_SQRT_2);
        assert_eq!(super::f64::FRAC_2_SQRT_PI, core::f64::consts::FRAC_2_SQRT_PI);
        assert_eq!(super::f64::FRAC_1_PI, core::f64::consts::FRAC_1_PI);
        assert_eq!(super::f64::FRAC_2_PI, core::f64::consts::FRAC_2_PI);
        assert_eq!(super::f64::FRAC_PI_2, core::f64::consts::FRAC_PI_2);
        assert_eq!(super::f64::FRAC_PI_3, core::f64::consts::FRAC_PI_3);
        assert_eq!(super::f64::FRAC_PI_4, core::f64::consts::FRAC_PI_4);
        assert_eq!(super::f64::FRAC_PI_6, core::f64::consts::FRAC_PI_6);
        assert_eq!(super::f64::FRAC_PI_8, core::f64::consts::FRAC_PI_8);
    }
}
