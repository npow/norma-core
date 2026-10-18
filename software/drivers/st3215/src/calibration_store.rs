//! Keep frozen calibration outside queue history, at ./calibration/<serial>.meta.
use crate::st3215_proto::{MetaEnvelope, MetaEnvelopeType};
use prost::Message;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

fn meta_path(directory: &Path, serial: &str) -> io::Result<PathBuf> {
    if serial.is_empty()
        || !serial
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid bus serial",
        ));
    }
    Ok(directory.join(format!("{serial}.meta")))
}

pub(crate) fn persist(directory: &Path, envelope: &MetaEnvelope) -> io::Result<()> {
    let path = meta_path(directory, &envelope.bus_serial)?;
    match MetaEnvelopeType::try_from(envelope.r#type) {
        Ok(MetaEnvelopeType::MetResetCalibration) => match std::fs::remove_file(path) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        },
        Ok(MetaEnvelopeType::MetFreezeCalibration) => {
            std::fs::create_dir_all(directory)?;
            let temporary = directory.join(format!(
                ".{}-{}-{}.tmp",
                envelope.bus_serial,
                std::process::id(),
                NEXT_FILE.fetch_add(1, Ordering::Relaxed),
            ));
            let result = (|| {
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&temporary)?;
                file.write_all(&envelope.encode_to_vec())?;
                file.sync_all()?;
                std::fs::rename(&temporary, &path)
            })();
            if result.is_err() {
                let _ = std::fs::remove_file(temporary);
            }
            result
        }
        _ => Ok(()),
    }
}

pub(crate) fn load(directory: &Path, serial: &str) -> Option<MetaEnvelope> {
    let path = meta_path(directory, serial).ok()?;
    let bytes = std::fs::read(&path).ok()?;
    match MetaEnvelope::decode(bytes.as_slice()) {
        Ok(env)
            if env.bus_serial == serial
                && env.r#type == MetaEnvelopeType::MetFreezeCalibration as i32 =>
        {
            Some(env)
        }
        _ => {
            log::warn!("Invalid frozen calibration file {:?}; ignoring", path);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::st3215_proto::MotorArc;

    struct TestDirectory(PathBuf);
    impl TestDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "norma-calibration-{}-{}",
                std::process::id(),
                NEXT_FILE.fetch_add(1, Ordering::Relaxed),
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn frozen() -> MetaEnvelope {
        MetaEnvelope {
            bus_serial: "5B3D045222".into(),
            r#type: MetaEnvelopeType::MetFreezeCalibration as i32,
            arcs: vec![MotorArc {
                motor_id: 1,
                min_angle: 1000,
                max_angle: 3000,
                range_freezed: true,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn freeze_replace_and_reset_preserve_exact_bounds() {
        let dir = TestDirectory::new();
        let mut envelope = frozen();
        persist(&dir.0, &envelope).unwrap();
        assert_eq!(load(&dir.0, &envelope.bus_serial), Some(envelope.clone()));
        envelope.arcs[0].min_angle = 1100;
        persist(&dir.0, &envelope).unwrap();
        assert_eq!(load(&dir.0, &envelope.bus_serial), Some(envelope.clone()));
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 1);
        envelope.r#type = MetaEnvelopeType::MetResetCalibration as i32;
        persist(&dir.0, &envelope).unwrap();
        assert!(load(&dir.0, &envelope.bus_serial).is_none());
        persist(&dir.0, &envelope).unwrap();
    }

    #[test]
    fn invalid_or_unrelated_records_do_not_restore_calibration() {
        let dir = TestDirectory::new();
        let mut envelope = frozen();
        let path = meta_path(&dir.0, &envelope.bus_serial).unwrap();
        std::fs::write(&path, b"corrupt protobuf").unwrap();
        assert!(load(&dir.0, &envelope.bus_serial).is_none());
        std::fs::write(
            &path,
            MetaEnvelope {
                bus_serial: "another-bus".into(),
                ..envelope.clone()
            }
            .encode_to_vec(),
        )
        .unwrap();
        assert!(load(&dir.0, &envelope.bus_serial).is_none());
        envelope.r#type = MetaEnvelopeType::MetResetCalibration as i32;
        std::fs::write(&path, envelope.encode_to_vec()).unwrap();
        assert!(load(&dir.0, &envelope.bus_serial).is_none());
        envelope.r#type = MetaEnvelopeType::MetUnknown as i32;
        persist(&dir.0, &envelope).unwrap();
        assert!(load(&dir.0, &envelope.bus_serial).is_none());
    }

    #[test]
    fn serial_cannot_escape_calibration_directory() {
        let dir = TestDirectory::new();
        for serial in ["", "../outside", "/absolute", "a/b", "a\\b"] {
            let mut envelope = frozen();
            envelope.bus_serial = serial.into();
            assert!(persist(&dir.0, &envelope).is_err());
            assert!(load(&dir.0, serial).is_none());
        }
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
    }
}
