//! Exact semantic versions (`major.minor.patch[-pre]`).
//!
//! Jax pins exact versions everywhere that matters, so there is deliberately
//! no range syntax: anything that looks like a range is rejected with an
//! explanation.

use std::cmp::Ordering;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub pre: Option<String>,
}

impl Version {
    pub fn new(major: u64, minor: u64, patch: u64) -> Version {
        Version {
            major,
            minor,
            patch,
            pre: None,
        }
    }

    pub fn parse(s: &str) -> Result<Version, String> {
        let s = s.trim();
        if s.is_empty() {
            return Err("version is empty".into());
        }
        if s.starts_with(|c: char| "^~<>=*vV".contains(c)) || s.contains(',') || s.contains('*') || s.contains(' ') {
            return Err(format!(
                "`{s}` is not an exact version; use something like `1.4.2` (version ranges are not allowed)"
            ));
        }
        let (core, pre) = match s.split_once('-') {
            Some((c, p)) => (c, Some(p.to_string())),
            None => (s, None),
        };
        if let Some(p) = &pre {
            if p.is_empty() || !p.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-') {
                return Err(format!("`{s}` has an invalid pre-release part"));
            }
        }
        let parts: Vec<&str> = core.split('.').collect();
        if parts.len() != 3 {
            return Err(format!("`{s}` is not an exact version; expected `major.minor.patch`"));
        }
        let mut nums = [0u64; 3];
        for (i, p) in parts.iter().enumerate() {
            nums[i] = p
                .parse::<u64>()
                .map_err(|_| format!("`{s}` is not a valid version: `{p}` is not a number"))?;
        }
        Ok(Version {
            major: nums[0],
            minor: nums[1],
            patch: nums[2],
            pre,
        })
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(p) = &self.pre {
            write!(f, "-{p}")?;
        }
        Ok(())
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Version) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Version) -> Ordering {
        (self.major, self.minor, self.patch)
            .cmp(&(other.major, other.minor, other.patch))
            .then_with(|| match (&self.pre, &other.pre) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Greater,
                (Some(_), None) => Ordering::Less,
                (Some(a), Some(b)) => a.cmp(b),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_order() {
        let a = Version::parse("1.4.2").unwrap();
        assert_eq!(a.to_string(), "1.4.2");
        assert!(Version::parse("1.10.0").unwrap() > Version::parse("1.9.9").unwrap());
        assert!(Version::parse("1.0.0-beta").unwrap() < Version::parse("1.0.0").unwrap());
    }

    #[test]
    fn rejects_ranges() {
        for bad in ["^1.2.3", "~1.2", ">=1.0.0", "1.2", "1.*.0", "latest", "1.2.x"] {
            assert!(Version::parse(bad).is_err(), "{bad}");
        }
    }
}
