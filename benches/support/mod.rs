#![allow(dead_code)]

use std::{fs, process::Command, sync::Once, time::Duration};

use criterion::{BenchmarkGroup, Throughput, measurement::WallTime};

pub(crate) const VECTOR_SIZES: [usize; 4] = [1 << 10, 1 << 14, 1 << 18, 1 << 20];
pub(crate) const SQUARE_MATRIX_SIDES: [usize; 4] = [32, 128, 512, 1024];
pub(crate) const DENSE_MATMUL_SIDES: [usize; 3] = [32, 64, 128];
pub(crate) const DENSE_MATMUL_RECT_SHAPES: [(usize, usize); 3] =
    [(128, 64), (512, 128), (2048, 256)];
pub(crate) const FACTORIZATION_SQUARE_SIZES: [usize; 3] = [16, 32, 64];
pub(crate) const FACTORIZATION_TALL_SHAPES: [(usize, usize); 3] = [(32, 16), (64, 32), (128, 64)];

const LARGE_INPUT_THRESHOLD: usize = 1 << 18;
const DEFAULT_SAMPLE_SIZE: usize = 100;
const LARGE_INPUT_SAMPLE_SIZE: usize = 50;
const DEFAULT_MEASUREMENT_SECS: u64 = 5;
const LARGE_INPUT_MEASUREMENT_SECS: u64 = 20;
const DEFAULT_WARM_UP_SECS: u64 = 3;

static BENCHMARK_METADATA_REPORTED: Once = Once::new();

pub(crate) fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>, work_items: usize) {
    group.throughput(Throughput::Elements(work_items as u64));
    configure_timing(group, work_items);
}

pub(crate) fn configure_timing(group: &mut BenchmarkGroup<'_, WallTime>, work_items: usize) {
    report_metadata();
    group.warm_up_time(Duration::from_secs(DEFAULT_WARM_UP_SECS));

    if work_items >= LARGE_INPUT_THRESHOLD {
        group.sample_size(LARGE_INPUT_SAMPLE_SIZE);
        group.measurement_time(Duration::from_secs(LARGE_INPUT_MEASUREMENT_SECS));
    } else {
        group.sample_size(DEFAULT_SAMPLE_SIZE);
        group.measurement_time(Duration::from_secs(DEFAULT_MEASUREMENT_SECS));
    }
}

fn report_metadata() {
    BENCHMARK_METADATA_REPORTED.call_once(|| {
        println!("atlas benchmark metadata:");
        println!("  os: {}", system_description());
        println!("  cpu: {}", cpu_description());
        println!("  ram: {}", memory_description());
        println!(
            "  rust: {}",
            command_output("rustc", &["--version"]).unwrap_or_else(|| "unknown".into())
        );
        println!(
            "  profile: bench ({})",
            if cfg!(debug_assertions) { "debug" } else { "optimized" }
        );
        println!(
            "  timing: warm-up={}s; default={} samples/{}s; large (>= {} work items)={} samples/{}s",
            DEFAULT_WARM_UP_SECS,
            DEFAULT_SAMPLE_SIZE,
            DEFAULT_MEASUREMENT_SECS,
            LARGE_INPUT_THRESHOLD,
            LARGE_INPUT_SAMPLE_SIZE,
            LARGE_INPUT_MEASUREMENT_SECS,
        );
    });
}

fn system_description() -> String {
    command_output("uname", &["-srm"]).unwrap_or_else(|| {
        format!(
            "{} {} ({})",
            std::env::consts::OS,
            std::env::consts::ARCH,
            std::env::consts::FAMILY
        )
    })
}

fn cpu_description() -> String {
    command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
        .or_else(|| {
            fs::read_to_string("/proc/cpuinfo").ok()?.lines().find_map(|line| {
                line.split_once(':')
                    .filter(|(name, _)| name.trim() == "model name")
                    .map(|(_, value)| value.trim().to_owned())
            })
        })
        .unwrap_or_else(|| {
            format!(
                "{} logical cores",
                std::thread::available_parallelism().map_or(0, |count| count.get())
            )
        })
}

fn memory_description() -> String {
    memory_bytes().map_or_else(
        || "unknown".into(),
        |bytes| format!("{:.1} GiB", bytes as f64 / 2_f64.powi(30)),
    )
}

fn memory_bytes() -> Option<u64> {
    command_output("sysctl", &["-n", "hw.memsize"]).and_then(|bytes| bytes.parse().ok()).or_else(
        || {
            fs::read_to_string("/proc/meminfo").ok()?.lines().find_map(|line| {
                line.strip_prefix("MemTotal:")?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()?
                    .checked_mul(1_024)
            })
        },
    )
}

fn command_output(command: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new(command).args(arguments).output().ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

pub(crate) fn square_label(side: usize) -> String {
    format!("{side}x{side}")
}

pub(crate) fn rect_label(rows: usize, cols: usize) -> String {
    format!("{rows}x{cols}")
}
