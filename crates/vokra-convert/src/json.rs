//! A minimal, dependency-free JSON parser.
//!
//! Just enough to read a safetensors header (objects, arrays, strings,
//! integers, floats, booleans, null). Numbers are parsed with the
//! locale-independent [`str::parse`] — never `strtod` (NFR-RL-01).
//!
//! This lives in `vokra-convert` (the offline tool), never in a runtime
//! crate, so it does not widen the runtime dependency surface (FR-LD-05).

use std::fmt;

/// A decoded JSON value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum JsonValue {
    /// JSON `null`.
    Null,
    /// JSON boolean.
    Bool(bool),
    /// A JSON number with no fractional/exponent part.
    Int(i64),
    /// A JSON number with a fractional or exponent part.
    Float(f64),
    /// JSON string (escapes decoded).
    Str(String),
    /// JSON array.
    Array(Vec<JsonValue>),
    /// JSON object, key order preserved.
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    /// Returns the object entries, or `None` if this is not an object.
    pub(crate) fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            Self::Object(entries) => Some(entries),
            _ => None,
        }
    }

    /// Returns the array elements, or `None` if this is not an array.
    pub(crate) fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    /// Returns the string payload, or `None` if this is not a string.
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the value as `u64` if it is a non-negative integer.
    pub(crate) fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Int(i) if *i >= 0 => Some(*i as u64),
            _ => None,
        }
    }

    /// Looks up a key in an object value.
    pub(crate) fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }
}

/// A JSON parse error with the byte offset at which it occurred.
#[derive(Debug)]
pub(crate) struct JsonError {
    /// Byte offset into the input where parsing failed.
    pub(crate) offset: usize,
    /// Human-readable reason.
    pub(crate) reason: &'static str,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "JSON parse error at byte {}: {}",
            self.offset, self.reason
        )
    }
}

impl std::error::Error for JsonError {}

/// Parses a complete JSON document, rejecting trailing non-whitespace.
pub(crate) fn parse(input: &[u8]) -> Result<JsonValue, JsonError> {
    let mut p = Parser { input, pos: 0 };
    p.skip_ws();
    let value = p.parse_value()?;
    p.skip_ws();
    if p.pos != input.len() {
        return Err(p.err("trailing data after JSON document"));
    }
    Ok(value)
}

struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn err(&self, reason: &'static str) -> JsonError {
        JsonError {
            offset: self.pos,
            reason,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, JsonError> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(JsonValue::Str(self.parse_string()?)),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'n') => self.parse_null(),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            _ => Err(self.err("unexpected token")),
        }
    }

    fn expect(&mut self, byte: u8, reason: &'static str) -> Result<(), JsonError> {
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(reason))
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, JsonError> {
        self.expect(b'{', "expected '{'")?;
        let mut entries = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(b':', "expected ':'")?;
            let value = self.parse_value()?;
            entries.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
        Ok(JsonValue::Object(entries))
    }

    fn parse_array(&mut self) -> Result<JsonValue, JsonError> {
        self.expect(b'[', "expected '['")?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            let value = self.parse_value()?;
            items.push(value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
        Ok(JsonValue::Array(items))
    }

    fn parse_string(&mut self) -> Result<String, JsonError> {
        self.expect(b'"', "expected '\"'")?;
        // Accumulate raw bytes so multi-byte UTF-8 sequences (whose lead and
        // continuation bytes arrive one at a time) are reassembled correctly.
        let mut bytes: Vec<u8> = Vec::new();
        loop {
            let b = self.peek().ok_or_else(|| self.err("unterminated string"))?;
            self.pos += 1;
            match b {
                b'"' => break,
                b'\\' => {
                    let esc = self.peek().ok_or_else(|| self.err("unterminated escape"))?;
                    self.pos += 1;
                    match esc {
                        b'"' => bytes.push(b'"'),
                        b'\\' => bytes.push(b'\\'),
                        b'/' => bytes.push(b'/'),
                        b'b' => bytes.push(0x08),
                        b'f' => bytes.push(0x0C),
                        b'n' => bytes.push(b'\n'),
                        b'r' => bytes.push(b'\r'),
                        b't' => bytes.push(b'\t'),
                        b'u' => {
                            let cp = self.parse_unicode_escape()?;
                            let mut buf = [0u8; 4];
                            bytes.extend_from_slice(cp.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return Err(self.err("invalid escape")),
                    }
                }
                // A raw control byte is invalid JSON; bytes >= 0x20 (including
                // UTF-8 lead/continuation bytes) are copied verbatim.
                0x00..=0x1F => return Err(self.err("control character in string")),
                other => bytes.push(other),
            }
        }
        String::from_utf8(bytes).map_err(|_| self.err("invalid UTF-8 in string"))
    }

    fn parse_unicode_escape(&mut self) -> Result<char, JsonError> {
        let first = self.read_hex4()?;
        // Handle a UTF-16 surrogate pair (\uD800-\uDBFF followed by \uDC00-).
        if (0xD800..=0xDBFF).contains(&first) {
            if self.peek() != Some(b'\\') {
                return Err(self.err("expected low surrogate"));
            }
            self.pos += 1;
            if self.peek() != Some(b'u') {
                return Err(self.err("expected low surrogate"));
            }
            self.pos += 1;
            let second = self.read_hex4()?;
            if !(0xDC00..=0xDFFF).contains(&second) {
                return Err(self.err("invalid low surrogate"));
            }
            let c = 0x1_0000 + ((first - 0xD800) << 10) + (second - 0xDC00);
            char::from_u32(c).ok_or_else(|| self.err("invalid code point"))
        } else {
            char::from_u32(first).ok_or_else(|| self.err("invalid code point"))
        }
    }

    fn read_hex4(&mut self) -> Result<u32, JsonError> {
        if self.pos + 4 > self.input.len() {
            return Err(self.err("truncated \\u escape"));
        }
        let hex = &self.input[self.pos..self.pos + 4];
        let s = std::str::from_utf8(hex).map_err(|_| self.err("invalid \\u escape"))?;
        let v = u32::from_str_radix(s, 16).map_err(|_| self.err("invalid \\u escape"))?;
        self.pos += 4;
        Ok(v)
    }

    fn parse_bool(&mut self) -> Result<JsonValue, JsonError> {
        if self.input[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(JsonValue::Bool(true))
        } else if self.input[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err(self.err("invalid literal"))
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, JsonError> {
        if self.input[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err(self.err("invalid literal"))
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue, JsonError> {
        let start = self.pos;
        let mut is_float = false;
        while let Some(b) = self.peek() {
            match b {
                b'0'..=b'9' | b'-' | b'+' => self.pos += 1,
                b'.' | b'e' | b'E' => {
                    is_float = true;
                    self.pos += 1;
                }
                _ => break,
            }
        }
        let text = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|_| self.err("invalid number"))?;
        if is_float {
            text.parse::<f64>()
                .map(JsonValue::Float)
                .map_err(|_| self.err("invalid float"))
        } else {
            match text.parse::<i64>() {
                Ok(i) => Ok(JsonValue::Int(i)),
                // Fall back to float for integers that overflow i64.
                Err(_) => text
                    .parse::<f64>()
                    .map(JsonValue::Float)
                    .map_err(|_| self.err("invalid integer")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_object_with_mixed_values() {
        let doc = br#"{ "a": 1, "b": [2, 3], "c": "x/y", "d": true, "e": null, "f": 1.5 }"#;
        let v = parse(doc).expect("valid json");
        assert_eq!(v.get("a").and_then(JsonValue::as_u64), Some(1));
        assert_eq!(
            v.get("b").and_then(JsonValue::as_array).map(<[_]>::len),
            Some(2)
        );
        assert_eq!(v.get("c").and_then(JsonValue::as_str), Some("x/y"));
        assert_eq!(v.get("d"), Some(&JsonValue::Bool(true)));
        assert_eq!(v.get("e"), Some(&JsonValue::Null));
        assert_eq!(v.get("f"), Some(&JsonValue::Float(1.5)));
    }

    #[test]
    fn parses_nested_and_empty() {
        assert_eq!(parse(b"{}").unwrap(), JsonValue::Object(vec![]));
        assert_eq!(parse(b"[]").unwrap(), JsonValue::Array(vec![]));
        let v = parse(br#"{"shape":[1,2,3],"data_offsets":[0,24]}"#).unwrap();
        let shape: Vec<u64> = v
            .get("shape")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect();
        assert_eq!(shape, vec![1, 2, 3]);
    }

    #[test]
    fn decodes_string_escapes() {
        let v = parse(r#""a\n\t\"\\\/Aé""#.as_bytes()).unwrap();
        assert_eq!(v.as_str(), Some("a\n\t\"\\/Aé"));
    }

    #[test]
    fn rejects_trailing_data() {
        assert!(parse(br#"{} junk"#).is_err());
    }

    #[test]
    fn rejects_unterminated_string() {
        assert!(parse(br#""abc"#).is_err());
    }

    #[test]
    fn decodes_bmp_unicode_escapes() {
        // é is 'é' (U+00E9); A is 'A' — both Basic-Multilingual-Plane.
        assert_eq!(parse(br#""\u00e9""#).unwrap().as_str(), Some("é"));
        assert_eq!(parse(br#""\u0041""#).unwrap().as_str(), Some("A"));
    }

    #[test]
    fn decodes_utf16_surrogate_pair() {
        // U+1F600 😀 is encoded as the UTF-16 surrogate pair 😀; the
        // decoder must reconstruct 0x10000 + ((hi-0xD800)<<10) + (lo-0xDC00).
        assert_eq!(parse(br#""\uD83D\uDE00""#).unwrap().as_str(), Some("😀"));
    }

    #[test]
    fn rejects_bad_surrogates_and_invalid_hex() {
        // Lone high surrogate with no following low surrogate.
        assert!(parse(br#""\uD800""#).is_err());
        // High surrogate followed by a non-low-surrogate code point (A).
        assert!(parse(br#""\uD800A""#).is_err());
        // Non-hex digits after \u.
        assert!(parse(br#""\uZZZZ""#).is_err());
        // Fewer than four hex digits before the closing quote (truncated).
        assert!(parse(br#""\uD8""#).is_err());
    }

    #[test]
    fn parses_number_edge_cases() {
        assert_eq!(parse(b"-5").unwrap(), JsonValue::Int(-5));
        assert_eq!(parse(b"1e3").unwrap(), JsonValue::Float(1000.0));
        // An integer literal that overflows i64 falls back to Float.
        assert!(matches!(
            parse(b"99999999999999999999").unwrap(),
            JsonValue::Float(_)
        ));
    }
}
